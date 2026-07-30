/*
 * Minimal host driver: dlopen the cdylib and drive it exactly as the JVM will
 * over Panama FFM -- symbol lookup by name, opaque handle, integer status codes.
 *
 * Proves the artifact is loadable and callable from outside Rust, which the Rust
 * integration tests (which link the rlib) cannot show.
 *
 * Build:
 *   cc -o host_smoke host_smoke.c -ldl
 *   ./host_smoke /path/to/libsteel_worldgen_ffi.so
 */

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int32_t x;
    int32_t z;
} SwgChunkPos;

typedef struct {
    const char *generator;
    int64_t seed;
    uint32_t threads;
} SwgWorldConfig;

typedef uint32_t (*fn_abi_version)(void);
typedef int32_t (*fn_runtime_init)(void);
typedef int32_t (*fn_world_open)(const SwgWorldConfig *, uint64_t *);
typedef int32_t (*fn_generate_batch)(uint64_t, const SwgChunkPos *, size_t, uint32_t, uint32_t,
                                     uint8_t *, size_t, size_t *);
typedef int32_t (*fn_world_close)(uint64_t);
typedef const char *(*fn_last_error)(void);

/* Chunk status indices, matching Steel's ChunkStatus enum. */
#define STATUS_FEATURES 7
#define STATUS_FULL 11

static void *must_sym(void *lib, const char *name) {
    void *sym = dlsym(lib, name);
    if (!sym) {
        fprintf(stderr, "FAIL: missing symbol %s: %s\n", name, dlerror());
        exit(1);
    }
    return sym;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <libsteel_worldgen_ffi.so>\n", argv[0]);
        return 2;
    }

    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "FAIL: dlopen: %s\n", dlerror());
        return 1;
    }

    fn_abi_version abi_version = (fn_abi_version)must_sym(lib, "swg_abi_version");
    fn_runtime_init runtime_init = (fn_runtime_init)must_sym(lib, "swg_runtime_init");
    fn_world_open world_open = (fn_world_open)must_sym(lib, "swg_world_open");
    fn_generate_batch generate_batch = (fn_generate_batch)must_sym(lib, "swg_generate_batch");
    fn_world_close world_close = (fn_world_close)must_sym(lib, "swg_world_close");
    fn_last_error last_error = (fn_last_error)must_sym(lib, "swg_last_error");

    uint32_t abi = abi_version();
    printf("abi_version = %u\n", abi);
    if (abi != 2) {
        fprintf(stderr, "FAIL: unexpected ABI version %u\n", abi);
        return 1;
    }

    int32_t status = runtime_init();
    printf("runtime_init = %d\n", status);
    if (status != 0) {
        fprintf(stderr, "FAIL: runtime_init: %s\n", last_error());
        return 1;
    }

    SwgWorldConfig config = {.generator = "minecraft:overworld", .seed = 424242, .threads = 16};
    uint64_t world = 0;
    status = world_open(&config, &world);
    printf("world_open = %d (handle %llu)\n", status, (unsigned long long)world);
    if (status != 0) {
        fprintf(stderr, "FAIL: world_open: %s\n", last_error());
        return 1;
    }

    /*
     * A square of chunks, driven all the way to Full, in batches.
     *
     * Timing includes the snapshot encode but not a host-side decode, since this
     * harness does not build chunk objects. Treat it as an upper bound.
     */
    int32_t side = (argc > 2) ? atoi(argv[2]) : 4;
    const int32_t batch_side = (argc > 3) ? atoi(argv[3]) : 4;
    SwgChunkPos *positions = malloc(sizeof(SwgChunkPos) * (size_t)batch_side * (size_t)batch_side);

    size_t capacity = 64u << 20;
    uint8_t *snapshot = malloc(capacity);
    size_t needed = 0;

    struct timespec started, finished;
    clock_gettime(CLOCK_MONOTONIC, &started);

    size_t generated = 0;
    size_t total_bytes = 0;
    for (int32_t bx = 0; bx < side; bx += batch_side) {
        for (int32_t bz = 0; bz < side; bz += batch_side) {
            size_t count = 0;
            for (int32_t x = bx; x < bx + batch_side && x < side; x++) {
                for (int32_t z = bz; z < bz + batch_side && z < side; z++) {
                    positions[count].x = x;
                    positions[count].z = z;
                    count++;
                }
            }

            status = generate_batch(world, positions, count, STATUS_FULL, 0, snapshot, capacity,
                                    &needed);
            if (status == -7) { /* BufferTooSmall: grow and retry */
                free(snapshot);
                capacity = needed;
                snapshot = malloc(capacity);
                status = generate_batch(world, positions, count, STATUS_FULL, 0, snapshot, capacity,
                                        &needed);
            }
            if (status != 0) {
                fprintf(stderr, "FAIL: generate_batch: %s\n", last_error());
                return 1;
            }
            if (needed < 4 || memcmp(snapshot, "SWGS", 4) != 0) {
                fprintf(stderr, "FAIL: snapshot magic missing\n");
                return 1;
            }
            generated += count;
            total_bytes += needed;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &finished);
    double elapsed = (double)(finished.tv_sec - started.tv_sec) +
                     (double)(finished.tv_nsec - started.tv_nsec) / 1e9;
    printf("generate_batch: %zu chunks -> Full in %.3fs (%.1f chunks/s, incl. snapshot encode)\n",
           generated, elapsed, (double)generated / elapsed);
    printf("snapshot: %.1f KiB total, %.1f KiB per chunk\n", (double)total_bytes / 1024.0,
           (double)total_bytes / 1024.0 / (double)generated);

    /* An unknown handle must be reported, not crash. */
    status = generate_batch(999999, positions, 1, STATUS_FEATURES, 0, snapshot, capacity, &needed);
    printf("generate_batch(bad handle) = %d (%s)\n", status, last_error());
    if (status != -5) {
        fprintf(stderr, "FAIL: expected InvalidHandle (-5), got %d\n", status);
        return 1;
    }

    status = world_close(world);
    printf("world_close = %d\n", status);
    if (status != 0) {
        fprintf(stderr, "FAIL: world_close: %s\n", last_error());
        return 1;
    }

    free(positions);
    free(snapshot);
    dlclose(lib);
    printf("OK\n");
    return 0;
}
