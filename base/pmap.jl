# This file is a part of Julia. License is MIT: http://julialang.org/license


"""
    pgenerate([::WorkerPool], f, c...) -> iterator

Apply `f` to each element of `c` in parallel using available workers and tasks.

For multiple collection arguments, apply f elementwise.

Results are returned in order as they become available.

Note that `f` must be made available to all worker processes; see
[Code Availability and Loading Packages](:ref:`Code Availability
and Loading Packages <man-parallel-computing-code-availability>`)
for details.
"""
function pgenerate(p::WorkerPool, f, c; distributed=true, batch_size=1, on_error=nothing)

    # Don't do remote calls if there are no workers.
    if (length(p) == 0) || (length(p) == 1 && fetch(p.channel) == myid())
        distributed = false
    end

    # Don't do batching if not doing remote calls.
    if !distributed
        batch_size = 1
    end

    # If not batching, do simple remote call.
    if batch_size == 1
        if distributed
            f = remote(p, f)
        end

        f = handle_on_error(f, on_error)
        return AsyncGenerator(f, c)
    else
        batches = batchsplit(c, min_batch_count = length(p) * 3,
                                max_batch_size = batch_size)
        f = handle_on_error(f, on_error)
        f = remote(p, asyncmap_batch(f))
        return flatten(AsyncGenerator(f, batches))
    end
end

pgenerate(p::WorkerPool, f, c1, c...; kwargs...) = pgenerate(p, a->f(a...), zip(c1, c...); kwargs...)

pgenerate(f, c; kwargs...) = pgenerate(default_worker_pool(), f, c...; kwargs...)
pgenerate(f, c1, c...; kwargs...) = pgenerate(a->f(a...), zip(c1, c...); kwargs...)

function handle_on_error(f, on_error)
    if on_error != nothing
        return x -> begin
            try
                f(x)
            catch e
                on_error(e)
            end
        end
    else
        return f
    end
end

function asyncmap_batch(f)
    return batch -> asyncmap(f, batch)
end
"""
    pmap([::WorkerPool], f, c...; distributed=true, batch_size=1, on_error=nothing) -> collection

Transform collection `c` by applying `f` to each element using available
workers and tasks.

For multiple collection arguments, apply f elementwise.

Note that `f` must be made available to all worker processes; see
[Code Availability and Loading Packages](:ref:`Code Availability
and Loading Packages <man-parallel-computing-code-availability>`)
for details.

If a worker pool is not specified, all available workers, i.e., the default worker pool
is used.

By default, `pmap` distributes the computation over all specified workers. To use only the
local process and distribute over tasks, specifiy `distributed=false`

`pmap` can also use a mix of processes and tasks via the `batch_size` argument. For batch sizes
greater than 1, the collection is split into multiple batches, which are distributed across
workers. Each such batch is processed in parallel via tasks in each worker. The specified
`batch_size` is an upper limit, the actual size of batches may be smaller and is calculated
depending on the number of workers available and length of the collection.

Any error stops pmap from processing the remainder of the collection. To override this behavior
you can specify an error handling function via argument `on_error` which takes in a single argument, i.e.,
the exception. The function can stop the processing by rethrowing the error, or, to continue, return any value
which is then returned inline with the results to the caller.
"""
pmap(p::WorkerPool, f, c...; kwargs...) = collect(pgenerate(p, f, c...; kwargs...))


"""
    batchsplit(c; min_batch_count=1, max_batch_size=100) -> iterator

Split a collection into at least `min_batch_count` batches.

Equivalent to `partition(c, max_batch_size)` when `length(c) >> max_batch_size`.
"""
function batchsplit(c; min_batch_count=1, max_batch_size=100)
    if min_batch_count < 1
        throw(ArgumentError("min_batch_count must be ≥ 1, got $min_batch_count"))
    end

    if max_batch_size < 1
        throw(ArgumentError("max_batch_size must be ≥ 1, got $max_batch_size"))
    end

    # Split collection into batches, then peek at the first few batches
    batches = partition(c, max_batch_size)
    head, tail = head_and_tail(batches, min_batch_count)

    # If there are not enough batches, use a smaller batch size
    if length(head) < min_batch_count
        batch_size = max(1, div(sum(length, head), min_batch_count))
        return partition(collect(flatten(head)), batch_size)
    end

    return flatten((head, tail))
end
