//! A status cell with an attached queue of waiters for later statuses.
//!
//! Models one monotonically rising status together with the work waiting on it.
//! Registration and status raises are a single atomic operation each, so a
//! waiter can never be inserted for a status that has already been published,
//! and a raise can never miss a waiter registered concurrently with it.
//!
//! Intended for chunk generation, where a holder publishes statuses in order
//! and dependents wait for a particular one.

use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

use crossbeam::epoch::{self, Atomic, Owned};

/// Status value reserved to mean "this queue will accept no further waiters".
///
/// Cancellation has to be distinguishable from any real status, and callers
/// must be able to tell "already satisfied" from "never will be".
const CANCELLED_STATUS: u16 = u16::MAX;

/// Outcome of trying to wait for a status.
#[derive(Debug)]
pub enum WaitOutcome<T> {
    /// The status was already at or past the one waited for; the payload is
    /// handed straight back rather than queued.
    AlreadySatisfied(T),
    /// The payload is queued and will be returned by a later raise.
    Registered,
    /// The queue is cancelled and will never satisfy this wait.
    Cancelled(T),
}

struct Node<T> {
    data: ManuallyDrop<T>,
    next: *mut Node<T>,
    /// Satisfied once the queue's status exceeds this.
    waiter_status: u16,
}

/// A monotonic status with a queue of waiters for higher statuses.
pub struct AtomicWaitQueue<T> {
    /// Packs the current status into the high 16 bits and the waiter-list head
    /// into the low 48.
    ///
    /// Both must move together: registration has to observe a status and a list
    /// in the same instant, or a raise landing between the two could drop a
    /// waiter that will never be woken.
    head: AtomicU64,
    /// Nodes detached from `head`, freed only once no thread can still be
    /// looking at them.
    ///
    /// Without this the structure has an ABA hole: a detached node can be freed
    /// and the allocator can hand the same address back for a new node while
    /// another thread still holds the old packed word, whose compare-exchange
    /// would then succeed against a different list.
    retired: Atomic<RetiredList<T>>,
    _marker: PhantomData<T>,
}

struct RetiredList<T> {
    node: *mut Node<T>,
    next: Atomic<RetiredList<T>>,
}

// SAFETY: the queue owns its nodes and every access to them goes through the
// packed atomic or epoch-protected retirement, so sharing it across threads is
// sound exactly when the payload can itself move between threads.
unsafe impl<T: Send> Send for AtomicWaitQueue<T> {}
// SAFETY: as above; `&AtomicWaitQueue<T>` only ever exposes `T` by value to one
// thread at a time, through a successful compare-exchange.
unsafe impl<T: Send> Sync for AtomicWaitQueue<T> {}

const STATUS_SHIFT: u64 = 48;
const PTR_MASK: u64 = (1u64 << STATUS_SHIFT) - 1;

impl<T> AtomicWaitQueue<T> {
    /// Creates a queue at `initial_status` with no waiters.
    #[must_use]
    pub const fn new(initial_status: u16) -> Self {
        Self {
            head: AtomicU64::new((initial_status as u64) << STATUS_SHIFT),
            retired: Atomic::null(),
            _marker: PhantomData,
        }
    }

    #[inline]
    const fn unpack(packed: u64) -> (u16, *mut Node<T>) {
        (
            (packed >> STATUS_SHIFT) as u16,
            (packed & PTR_MASK) as *mut Node<T>,
        )
    }

    #[inline]
    fn pack(status: u16, node: *mut Node<T>) -> u64 {
        let address = node as u64;
        debug_assert_eq!(
            address & !PTR_MASK,
            0,
            "node address does not fit in {STATUS_SHIFT} bits; the packing assumes a 48-bit \
             user-space address space"
        );
        (u64::from(status) << STATUS_SHIFT) | (address & PTR_MASK)
    }

    /// Current status, or `None` once cancelled.
    #[must_use]
    pub fn status(&self) -> Option<u16> {
        let (status, _) = Self::unpack(self.head.load(Ordering::Acquire));
        (status != CANCELLED_STATUS).then_some(status)
    }

    /// Queues `payload` until the status rises past `wait_for`.
    ///
    /// Returns the payload immediately when the status is already past it, or
    /// when the queue is cancelled.
    pub fn wait(&self, wait_for: u16, payload: T) -> WaitOutcome<T> {
        // Answer from the status word alone where that settles it, which is the
        // common case: a dependent usually asks about a status that is already
        // published. Allocating first and freeing it again on the way out cost a
        // malloc/free pair per such call, and the design this primitive exists
        // for registers against every unmet dependency of every chunk.
        //
        // Racing with a publication is fine either way. Losing the race means
        // reaching the loop below and discovering it there; the loop re-reads
        // the status and reaches the same answer.
        let mut current = self.head.load(Ordering::Acquire);
        {
            let (status, _) = Self::unpack(current);
            if status == CANCELLED_STATUS {
                return WaitOutcome::Cancelled(payload);
            }
            if status > wait_for {
                return WaitOutcome::AlreadySatisfied(payload);
            }
        }

        let node = Box::into_raw(Box::new(Node {
            data: ManuallyDrop::new(payload),
            next: ptr::null_mut(),
            waiter_status: wait_for,
        }));

        loop {
            let (status, head) = Self::unpack(current);

            let reclaim = |outcome: fn(T) -> WaitOutcome<T>| {
                // SAFETY: `node` was just allocated here and has not been
                // published, so this is the only reference to it.
                let payload = unsafe {
                    let owned = Box::from_raw(node);
                    ManuallyDrop::into_inner(owned.data)
                };
                outcome(payload)
            };

            if status == CANCELLED_STATUS {
                return reclaim(WaitOutcome::Cancelled);
            }
            if status > wait_for {
                return reclaim(WaitOutcome::AlreadySatisfied);
            }

            // SAFETY: `node` is not yet published, so writing its link is
            // unsynchronized and cannot race.
            unsafe { (*node).next = head };

            match self.head.compare_exchange_weak(
                current,
                Self::pack(status, node),
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => return WaitOutcome::Registered,
                Err(actual) => current = actual,
            }
        }
    }

    /// Raises the status and hands every newly satisfied payload to `notify`.
    ///
    /// Waiters that need a later status are put back. A raise that does not
    /// exceed the current status does nothing, so racing raisers are safe in
    /// either order.
    pub fn advance_and_notify<F>(&self, new_status: u16, notify: F)
    where
        F: FnMut(T),
    {
        debug_assert_ne!(
            new_status, CANCELLED_STATUS,
            "use `cancel` rather than raising to the cancelled sentinel"
        );

        let detached = self.swap_in_status(new_status);
        if detached.is_null() {
            return;
        }
        self.drain(detached, new_status, notify);
    }

    /// Marks the queue as accepting no further waiters and hands back everything
    /// still queued.
    pub fn cancel<F>(&self, mut discard: F)
    where
        F: FnMut(T),
    {
        let detached = self.swap_in_status(CANCELLED_STATUS);
        if detached.is_null() {
            return;
        }
        // Everything is unsatisfied by definition, so nothing is re-queued.
        let mut node = detached;
        while !node.is_null() {
            // SAFETY: the list was detached atomically, so these nodes are ours.
            unsafe {
                let next = (*node).next;
                discard(ManuallyDrop::take(&mut (*node).data));
                drop(Box::from_raw(node));
                node = next;
            }
        }
    }

    /// Atomically publishes `new_status` and takes the whole waiter list.
    ///
    /// A raise to a status at or below the current one takes nothing and
    /// publishes nothing. Everything such a raise would satisfy was already
    /// satisfied by the status that is there -- a waiter needing less than the
    /// current status never joins the list, because `wait` hands it straight
    /// back -- and waiters needing more are still queued for whoever gets there.
    ///
    /// This is what lets concurrent raisers disagree about order. Callers guard
    /// with a load-then-raise that two threads can pass at once, so the raises
    /// can arrive in either order; without this the lower one would publish its
    /// status over the higher one's and the queue would go backwards.
    fn swap_in_status(&self, new_status: u16) -> *mut Node<T> {
        let mut current = self.head.load(Ordering::Acquire);
        loop {
            let (status, head) = Self::unpack(current);
            if status == CANCELLED_STATUS || new_status <= status {
                return ptr::null_mut();
            }

            match self.head.compare_exchange_weak(
                current,
                Self::pack(new_status, ptr::null_mut()),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return head,
                Err(actual) => current = actual,
            }
        }
    }

    /// Splits a detached list, notifying the satisfied and re-queueing the rest.
    fn drain<F>(&self, detached: *mut Node<T>, new_status: u16, mut notify: F)
    where
        F: FnMut(T),
    {
        let mut node = detached;
        let mut keep_head: *mut Node<T> = ptr::null_mut();
        let mut keep_tail: *mut Node<T> = ptr::null_mut();

        while !node.is_null() {
            // SAFETY: the list was detached atomically, so these nodes are ours
            // until we either retire them or re-publish them.
            unsafe {
                let next = (*node).next;
                (*node).next = ptr::null_mut();

                if new_status > (*node).waiter_status {
                    notify(ManuallyDrop::take(&mut (*node).data));
                    self.retire(node);
                } else if keep_head.is_null() {
                    keep_head = node;
                    keep_tail = node;
                } else {
                    (*keep_tail).next = node;
                    keep_tail = node;
                }
                node = next;
            }
        }

        if keep_head.is_null() {
            return;
        }

        // Prepend rather than replace: waiters registered while the list was
        // detached are in the current head and must not be lost.
        let mut current = self.head.load(Ordering::Acquire);
        loop {
            let (status, head) = Self::unpack(current);
            // SAFETY: `keep_tail` is ours until the exchange publishes it.
            unsafe { (*keep_tail).next = head };

            match self.head.compare_exchange_weak(
                current,
                Self::pack(status, keep_head),
                Ordering::Release,
                Ordering::Acquire,
            ) {
                Ok(_) => return,
                Err(actual) => current = actual,
            }
        }
    }

    /// Defers freeing a node until no thread can still hold the packed word
    /// that pointed at it.
    fn retire(&self, node: *mut Node<T>) {
        let guard = &epoch::pin();
        let entry = Owned::new(RetiredList {
            node,
            next: Atomic::null(),
        })
        .into_shared(guard);

        let mut current = self.retired.load(Ordering::Acquire, guard);
        loop {
            // SAFETY: `entry` was just allocated and is not yet published.
            unsafe { entry.deref().next.store(current, Ordering::Relaxed) };
            match self.retired.compare_exchange_weak(
                current,
                entry,
                Ordering::Release,
                Ordering::Acquire,
                guard,
            ) {
                Ok(_) => return,
                Err(error) => current = error.current,
            }
        }
    }
}

impl<T> Drop for AtomicWaitQueue<T> {
    fn drop(&mut self) {
        let (_, mut node) = Self::unpack(self.head.load(Ordering::Relaxed));
        while !node.is_null() {
            // SAFETY: `&mut self` means nothing else can reach these nodes. The
            // payload is `ManuallyDrop`, so dropping the box alone would leak it.
            unsafe {
                let next = (*node).next;
                ManuallyDrop::drop(&mut (*node).data);
                drop(Box::from_raw(node));
                node = next;
            }
        }

        let guard = &epoch::pin();
        let mut retired = self.retired.load(Ordering::Relaxed, guard);
        while !retired.is_null() {
            // SAFETY: as above, exclusive access. The payload was already taken
            // when the node was retired, so only the allocation remains.
            unsafe {
                let entry = retired.deref();
                drop(Box::from_raw(entry.node));
                let next = entry.next.load(Ordering::Relaxed, guard);
                drop(retired.into_owned());
                retired = next;
            }
        }
    }
}

impl<T> Default for AtomicWaitQueue<T> {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn waiting_past_the_current_status_hands_the_payload_straight_back() {
        let queue = AtomicWaitQueue::new(5);
        let outcome = queue.wait(4, "done");
        assert!(matches!(outcome, WaitOutcome::AlreadySatisfied("done")));
    }

    #[test]
    fn a_hand_back_leaves_nothing_registered() {
        // The status-only fast path returns without allocating a node. If it ever
        // published one anyway, the payload would be handed back here *and* again
        // by the next raise.
        let queue = AtomicWaitQueue::<&str>::new(5);
        assert!(matches!(
            queue.wait(3, "already past"),
            WaitOutcome::AlreadySatisfied("already past")
        ));

        let mut released = Vec::new();
        queue.advance_and_notify(6, |payload| released.push(payload));
        assert!(
            released.is_empty(),
            "the handed-back waiter must not also be queued: {released:?}"
        );
    }

    #[test]
    fn a_cancelled_queue_hands_back_without_registering() {
        let queue = AtomicWaitQueue::<&str>::new(0);
        queue.cancel(|_| {});
        assert!(matches!(
            queue.wait(9, "after cancel"),
            WaitOutcome::Cancelled("after cancel")
        ));
    }

    #[test]
    fn a_registered_waiter_is_released_only_once_its_status_is_passed() {
        let queue = AtomicWaitQueue::new(0);
        assert!(matches!(queue.wait(2, "at-2"), WaitOutcome::Registered));

        let mut woken = Vec::new();
        queue.advance_and_notify(2, |payload| woken.push(payload));
        assert!(woken.is_empty(), "status 2 does not pass a waiter for 2");

        queue.advance_and_notify(3, |payload| woken.push(payload));
        assert_eq!(woken, ["at-2"]);
    }

    #[test]
    fn a_raise_keeps_waiters_that_need_a_later_status() {
        let queue = AtomicWaitQueue::new(0);
        assert!(matches!(queue.wait(1, "early"), WaitOutcome::Registered));
        assert!(matches!(queue.wait(9, "late"), WaitOutcome::Registered));

        let mut woken = Vec::new();
        queue.advance_and_notify(5, |payload| woken.push(payload));
        assert_eq!(woken, ["early"]);

        woken.clear();
        queue.advance_and_notify(10, |payload| woken.push(payload));
        assert_eq!(woken, ["late"], "the late waiter survived the first raise");
    }

    #[test]
    fn cancelling_hands_back_every_queued_payload_and_refuses_new_ones() {
        let queue = AtomicWaitQueue::new(0);
        assert!(matches!(queue.wait(1, "a"), WaitOutcome::Registered));
        assert!(matches!(queue.wait(2, "b"), WaitOutcome::Registered));

        let mut discarded = Vec::new();
        queue.cancel(|payload| discarded.push(payload));
        discarded.sort_unstable();
        assert_eq!(discarded, ["a", "b"]);

        assert!(queue.status().is_none());
        assert!(matches!(queue.wait(1, "c"), WaitOutcome::Cancelled("c")));
    }

    #[test]
    fn dropping_the_queue_drops_queued_payloads() {
        struct CountsDrops(Arc<AtomicUsize>);
        impl Drop for CountsDrops {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::Relaxed);
            }
        }

        let drops = Arc::new(AtomicUsize::new(0));
        {
            let queue = AtomicWaitQueue::new(0);
            for _ in 0..3 {
                assert!(matches!(
                    queue.wait(1, CountsDrops(Arc::clone(&drops))),
                    WaitOutcome::Registered
                ));
            }
        }
        assert_eq!(
            drops.load(Ordering::Relaxed),
            3,
            "payloads still queued at drop must not leak"
        );
    }

    #[test]
    fn concurrent_waiters_are_each_woken_exactly_once() {
        const WAITERS: usize = 64;
        const RAISES: u16 = 16;

        let queue = Arc::new(AtomicWaitQueue::<usize>::new(0));
        let woken = Arc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            for i in 0..WAITERS {
                let queue = Arc::clone(&queue);
                let woken = Arc::clone(&woken);
                scope.spawn(move || {
                    let target = (i % RAISES as usize) as u16;
                    if let WaitOutcome::AlreadySatisfied(_) = queue.wait(target, i) {
                        woken.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
            for status in 1..=RAISES {
                let queue = Arc::clone(&queue);
                let woken = Arc::clone(&woken);
                scope.spawn(move || {
                    queue.advance_and_notify(status, |_| {
                        woken.fetch_add(1, Ordering::Relaxed);
                    });
                });
            }
        });

        // Drain anything still queued below the final status.
        queue.advance_and_notify(RAISES + 1, |_| {
            woken.fetch_add(1, Ordering::Relaxed);
        });

        assert_eq!(
            woken.load(Ordering::Relaxed),
            WAITERS,
            "every waiter is accounted for exactly once, with no lost wakeups"
        );
    }
}
