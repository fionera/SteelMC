//! `ChunkGenerationTask` handles the generation process for chunks.
use std::{
    future::Future,
    mem,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use futures::future::join_all;
use rayon::ThreadPool;
use steel_utils::{ChunkPos, locks::SyncMutex};
use tokio_util::sync::CancellationToken;

use crate::chunk::{
    chunk_holder::ChunkHolder,
    chunk_map::ChunkMap,
    chunk_pyramid::GENERATION_PYRAMID,
    static_cache_2d::resolve_halo_at,
    status::ChunkStatus,
};

// Re-exported from its own module so the ~10 `use
// crate::chunk::chunk_generation_task::StaticCache2D` sites across `worldgen`
// keep working.
pub use crate::chunk::static_cache_2d::StaticCache2D;

/// A pinned future representing a neighbor's readiness.
pub type NeighborReady = Pin<Box<dyn Future<Output = Option<()>> + Send + Sync>>;

/// A task responsible for driving a chunk to a target status.
///
/// It works in form of layers. Imagine a pyramid, to get to the top you first need to generate the base layer. And so on.
/// This works in the same way but with chunk dependencies.
///
/// This is achieved using the Generation Pyramid and Loading Pyramid.
///
/// To make sure a chunk is only put through a stage once it uses an atomic with a CAS operation. Loading Pyramid must also be advanced with noop functions so this atomic can be driven forward.
pub struct ChunkGenerationTask {
    /// The chunk map associated with this task.
    pub chunk_map: Arc<ChunkMap>,
    /// The chunk position.
    pub pos: ChunkPos,
    /// The target generation status.
    pub target_status: ChunkStatus,
    /// The status scheduled for generation. Protected by a mutex for safe concurrent access.
    pub scheduled_status: SyncMutex<Option<ChunkStatus>>,
    /// Cancellation token — cancelled when this task should stop.
    pub cancel_token: CancellationToken,
    /// Cheap cancellation flag for scheduler-side filtering.
    cancelled: AtomicBool,
    /// Futures for neighbors. Protected by a mutex.
    pub neighbor_ready: SyncMutex<Vec<NeighborReady>>,
    /// Cache of required chunks.
    /// Radius over which this task's dependency halo must be resolved.
    worst_case_radius: i32,
    /// Holder for the chunk this task is targeting.
    pub center_holder: Arc<ChunkHolder>,
    /// The thread pool to use for generation.
    pub thread_pool: Arc<ThreadPool>,
}

impl ChunkGenerationTask {
    /// Creates a new generation task.
    #[must_use]
    #[inline]
    #[expect(
        clippy::missing_panics_doc,
        reason = "panic is unreachable: ThreadPoolBuilder::build only fails on OS thread errors"
    )]
    pub fn new(
        pos: ChunkPos,
        target_status: ChunkStatus,
        chunk_map: Arc<ChunkMap>,
        thread_pool: Arc<ThreadPool>,
        cancel_token: CancellationToken,
    ) -> Self {
        let worst_case_radius = GENERATION_PYRAMID
            .get_step_to(target_status)
            .accumulated_dependencies
            .get_radius_of(ChunkStatus::Empty) as i32;

        let center_holder = chunk_map
            .chunks
            .read_sync(&pos, |_, chunk_holder| chunk_holder.clone())
            .expect("The chunkholder should be created by distance manager before the generation task is scheduled. This occurring means there is a bug in the distance manager or you called this yourself.");

        Self {
            chunk_map,
            pos,
            target_status,
            scheduled_status: SyncMutex::new(None),
            cancel_token,
            cancelled: AtomicBool::new(false),
            neighbor_ready: SyncMutex::new(Vec::new()),
            worst_case_radius,
            center_holder,
            thread_pool,
        }
    }

    /// Resolves the dependency halo this task will operate on.
    ///
    /// Deliberately not done in `new`. Task construction runs on the single
    /// scheduling-epoch thread, and a `Full` target resolves a radius-11 halo --
    /// 529 holder lookups and 529 `Arc` clones per chunk. Over a pregeneration
    /// that put tens of millions of map lookups on the one thread every chunk
    /// has to pass through to exist: it measured 44-55% of wall clock, with
    /// individual epochs stalling task creation for up to 2.3s while the
    /// generation pool drained. Resolving it here instead spreads the same work
    /// across the chunk runtime, where each task pays only for itself.
    ///
    /// Returns `None` when a halo holder has been unloaded since the task was
    /// scheduled, which construction could treat as impossible but this cannot:
    /// tickets may be dropped while the task waits in the pending queue.
    fn resolve_halo(&self) -> Option<Arc<StaticCache2D<Arc<ChunkHolder>>>> {
        resolve_halo_at(&self.chunk_map, self.pos, self.worst_case_radius)
    }

    /// Cancels this task by triggering the cancellation token.
    pub fn cancel(&self) {
        if !self.cancelled.swap(true, Ordering::AcqRel) {
            self.cancel_token.cancel();
        }
    }

    /// Returns whether this task has been explicitly cancelled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    /// Returns the holder for the chunk this task is targeting.
    pub(crate) const fn center_holder(&self) -> &Arc<ChunkHolder> {
        &self.center_holder
    }

    /// Schedules a chunk for a specific layer.
    ///
    /// # Panics
    /// Panics if generation is required but not expected.
    pub fn schedule_chunk_in_layer(
        &self,
        halo: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
        status: ChunkStatus,
        chunk_holder: &Arc<ChunkHolder>,
    ) -> bool {
        let published_status = chunk_holder.published_status();

        let generate;
        if let Some(published_status) = published_status {
            generate = status > published_status;
        } else {
            generate = true;
        }

        if !generate {
            // Already published at or past this status, so there is nothing to
            // run and nothing to wait for: `apply_step` would lose the
            // `claim_status_work` race and hand back a future that resolves on
            // its first poll.
            //
            // Skipping matters because a task schedules its entire accumulated
            // dependency neighbourhood at every layer -- 1,252 `apply_step`
            // calls for a `Full` target, of which the two widest layers are 529
            // chunks each -- while only the handful covering its own centre have
            // work left. Every one of the rest was allocating a boxed future,
            // cloning an `Arc<ChunkHolder>` and registering a `Notify` waiter to
            // observe a status that was already published. At ~4,000 chunks/s
            // that is over five million such calls per second, and it is why the
            // chunk runtime was consuming ~12 cores of pure bookkeeping.
            return true;
        }

        let pyramid = &GENERATION_PYRAMID;

        if let Some(future) = chunk_holder.apply_step(
            pyramid.get_step_to(status),
            &self.chunk_map,
            halo,
            self.thread_pool.clone(),
        ) {
            self.neighbor_ready.lock().push(future);
        } else {
            self.cancel();
        }

        true
    }

    /// Schedules tasks for the current layer's neighbors.
    pub fn schedule_layer(
        &self,
        halo: &Arc<StaticCache2D<Arc<ChunkHolder>>>,
        status: ChunkStatus,
    ) {
        let radius = self.get_radius_for_layer(status);
        // This for loop is inclusive, so if the radius is 0, we will only schedule the center chunk.
        for x in (self.pos.0.x - radius)..=(self.pos.0.x + radius) {
            for y in (self.pos.0.y - radius)..=(self.pos.0.y + radius) {
                let chunk_holder = halo.get(x, y);
                if self.is_cancelled()
                    || !self.schedule_chunk_in_layer(halo, status, chunk_holder)
                {
                    return;
                }
            }
        }
    }

    const fn get_radius_for_layer(&self, status: ChunkStatus) -> i32 {
        GENERATION_PYRAMID
            .get_step_to(self.target_status)
            .get_accumulated_radius_of(status) as i32
    }

    /// Schedules the next layer of generation dependencies.
    ///
    /// # Panics
    /// Panics if the schedule is invalid.
    pub fn schedule_next_layer(&self, halo: &Arc<StaticCache2D<Arc<ChunkHolder>>>) {
        let status_to_schedule = if self.scheduled_status.lock().is_none() {
            ChunkStatus::Empty
        } else {
            self.scheduled_status
                .lock()
                .expect("Scheduled status missing")
                .next()
                .expect("Next status missing")
        };

        self.schedule_layer(halo, status_to_schedule);
        self.scheduled_status.lock().replace(status_to_schedule);
    }

    /// Runs the generation task loop.
    pub async fn run(self: Arc<Self>) {
        let Some(halo) = self.resolve_halo() else {
            self.cancel();
            self.center_holder.clear_generation_task_if_current(&self);
            return;
        };

        loop {
            tokio::select! {
                () = self.cancel_token.cancelled() => break,
                () = self.wait_for_scheduled_layers() => {}
            }

            if *self.scheduled_status.lock() == Some(self.target_status) {
                break;
            }

            self.schedule_next_layer(&halo);
        }
        self.center_holder
            .clear_generation_task_if_current(&self);
    }

    /// Waits for all scheduled neighbor tasks to complete.
    pub async fn wait_for_scheduled_layers(&self) {
        // Collect all futures first to avoid locking the mutex during await
        let futures: Vec<_> = {
            let mut lock = self.neighbor_ready.lock();
            mem::take(&mut *lock)
        };

        if futures.is_empty() {
            return;
        }

        let results = join_all(futures).await;

        for result in results {
            if result.is_none() {
                self.cancel();
                break;
            }
        }
    }
}
