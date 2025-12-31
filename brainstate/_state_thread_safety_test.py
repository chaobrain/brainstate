# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Thread safety stress tests for hooks."""

import threading
import time
from unittest import TestCase
import jax.numpy as jnp
import brainstate 


class TestHookThreadSafety(TestCase):
    """Stress tests for thread safety of hook system."""

    def setUp(self):
        """Set up test fixtures."""
        brainstate.clear_state_hooks()

    def tearDown(self):
        """Clean up after tests."""
        brainstate.clear_state_hooks()

    def test_concurrent_state_reads_with_hooks(self):
        """Test multiple threads reading state with hooks concurrently."""
        state = brainstate.State(jnp.array([1, 2, 3]))
        read_count = {'count': 0}
        lock = threading.Lock()

        def count_reads(ctx):
            with lock:
                read_count['count'] += 1

        state.register_hook('read', count_reads)

        num_threads = 20
        reads_per_thread = 100

        def reader_thread():
            for _ in range(reads_per_thread):
                _ = state.value

        threads = [threading.Thread(target=reader_thread) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should be counted
        expected_count = num_threads * reads_per_thread
        self.assertEqual(read_count['count'], expected_count)

    def test_concurrent_state_writes_with_hooks(self):
        """Test multiple threads writing to state with hooks concurrently."""
        state = brainstate.State(jnp.array([0]))
        write_count = {'count': 0}
        lock = threading.Lock()

        def count_writes(ctx):
            with lock:
                write_count['count'] += 1

        state.register_hook('write_after', count_writes)

        num_threads = 20
        writes_per_thread = 50

        def writer_thread(thread_id):
            for i in range(writes_per_thread):
                state.value = jnp.array([thread_id * 1000 + i])

        threads = [threading.Thread(target=writer_thread, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All writes should be counted
        expected_count = num_threads * writes_per_thread
        self.assertEqual(write_count['count'], expected_count)

    def test_concurrent_hook_registration_during_execution(self):
        """Test registering hooks while hooks are executing."""
        state = brainstate.State(jnp.array([1, 2, 3]))
        execution_log = []
        log_lock = threading.Lock()

        def initial_hook(ctx):
            with log_lock:
                execution_log.append('initial')
            time.sleep(0.001)  # Small delay to increase chance of race

        state.register_hook('read', initial_hook)

        num_registrations = 10
        num_reads = 50
        stop_flag = threading.Event()

        def register_hooks():
            for i in range(num_registrations):
                state.register_hook('read', lambda ctx, i=i: execution_log.append(f'hook_{i}'))
                time.sleep(0.002)

        def read_values():
            while not stop_flag.is_set():
                _ = state.value
                time.sleep(0.001)

        # Start reader threads
        reader_threads = [threading.Thread(target=read_values) for _ in range(3)]
        for t in reader_threads:
            t.start()

        # Register new hooks while reading
        register_thread = threading.Thread(target=register_hooks)
        register_thread.start()
        register_thread.join()

        # Stop readers
        stop_flag.set()
        for t in reader_threads:
            t.join()

        # Should have executed without crashes
        self.assertGreater(len(execution_log), 0)

    def test_concurrent_hook_enable_disable(self):
        """Test enabling/disabling hooks concurrently."""
        state = brainstate.State(jnp.array([1, 2, 3]))

        handles = []
        for i in range(10):
            handle = state.register_hook('read', lambda ctx: None, name=f'hook_{i}')
            handles.append(handle)

        num_toggles = 100
        num_threads = 5

        def toggle_hooks():
            for _ in range(num_toggles):
                for handle in handles:
                    if handle.is_enabled():
                        handle.disable()
                    else:
                        handle.enable()
                time.sleep(0.0001)

        def read_values():
            for _ in range(num_toggles * 2):
                _ = state.value
                time.sleep(0.0001)

        toggle_threads = [threading.Thread(target=toggle_hooks) for _ in range(num_threads)]
        reader_threads = [threading.Thread(target=read_values) for _ in range(num_threads)]

        all_threads = toggle_threads + reader_threads
        for t in all_threads:
            t.start()
        for t in all_threads:
            t.join()

        # Should complete without crashes or deadlocks

    def test_concurrent_global_and_instance_hooks(self):
        """Test global and instance hooks executing concurrently."""
        global_count = {'count': 0}
        instance_count = {'count': 0}
        lock = threading.Lock()

        def global_hook(ctx):
            with lock:
                global_count['count'] += 1

        def instance_hook(ctx):
            with lock:
                instance_count['count'] += 1

        brainstate.register_state_hook('read', global_hook)

        num_states = 5
        states = [brainstate.State(jnp.array([i])) for i in range(num_states)]

        for state in states:
            state.register_hook('read', instance_hook)

        num_threads = 10
        reads_per_thread = 20

        def read_all_states():
            for _ in range(reads_per_thread):
                for state in states:
                    _ = state.value

        threads = [threading.Thread(target=read_all_states) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected_count = num_threads * reads_per_thread * num_states
        self.assertEqual(global_count['count'], expected_count)
        self.assertEqual(instance_count['count'], expected_count)

    def test_concurrent_write_transformations(self):
        """Test concurrent write transformations are safe."""
        state = brainstate.State(jnp.array([0.0]))

        def clip_hook(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = jnp.clip(input_val, -10.0, 10.0)

        def scale_hook(ctx):
            input_val = ctx.transformed_value if ctx.transformed_value is not None else ctx.value
            ctx.transformed_value = input_val * 0.5

        state.register_hook('write_before', clip_hook, priority=10)
        state.register_hook('write_before', scale_hook, priority=5)

        num_threads = 10
        writes_per_thread = 20

        def writer_thread(thread_id):
            for i in range(writes_per_thread):
                value = float(thread_id * 100 + i)
                state.value = jnp.array([value])

        threads = [threading.Thread(target=writer_thread, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Final value should be within bounds
        final_value = float(state.value[0])
        self.assertGreaterEqual(final_value, -10.0)
        self.assertLessEqual(final_value, 10.0)

    def test_race_condition_hook_removal(self):
        """Test removing hooks while they're being executed."""
        state = brainstate.State(jnp.array([1]))
        execution_count = {'count': 0}
        lock = threading.Lock()

        def slow_hook(ctx):
            with lock:
                execution_count['count'] += 1
            time.sleep(0.01)

        handles = []
        for i in range(5):
            handle = state.register_hook('read', slow_hook)
            handles.append(handle)

        stop_flag = threading.Event()

        def reader_thread():
            while not stop_flag.is_set():
                try:
                    _ = state.value
                except:
                    pass  # Ignore errors from hook removal
                time.sleep(0.005)

        def remover_thread():
            time.sleep(0.05)  # Let some reads happen first
            for handle in handles:
                handle.remove()
                time.sleep(0.01)
            stop_flag.set()

        reader_threads = [threading.Thread(target=reader_thread) for _ in range(3)]
        remover = threading.Thread(target=remover_thread)

        for t in reader_threads:
            t.start()
        remover.start()

        remover.join()
        for t in reader_threads:
            t.join()

        # Should complete without crashes
        self.assertGreater(execution_count['count'], 0)

    def test_multiple_states_concurrent_operations(self):
        """Test multiple states with hooks operating concurrently."""
        num_states = 10
        states = [brainstate.State(jnp.array([i])) for i in range(num_states)]

        counters = [{'reads': 0, 'writes': 0} for _ in range(num_states)]
        locks = [threading.Lock() for _ in range(num_states)]

        for i, state in enumerate(states):
            def make_read_hook(idx):
                def hook(ctx):
                    with locks[idx]:
                        counters[idx]['reads'] += 1
                return hook

            def make_write_hook(idx):
                def hook(ctx):
                    with locks[idx]:
                        counters[idx]['writes'] += 1
                return hook

            state.register_hook('read', make_read_hook(i))
            state.register_hook('write_after', make_write_hook(i))

        num_threads = 5
        ops_per_thread = 20

        def operate_on_states(thread_id):
            for i in range(ops_per_thread):
                state_idx = (thread_id + i) % num_states
                _ = states[state_idx].value
                states[state_idx].value = jnp.array([thread_id * 100 + i])

        threads = [threading.Thread(target=operate_on_states, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Verify all operations were counted
        total_reads = sum(c['reads'] for c in counters)
        total_writes = sum(c['writes'] for c in counters)
        expected_ops = num_threads * ops_per_thread
        self.assertEqual(total_reads, expected_ops)
        self.assertEqual(total_writes, expected_ops)


if __name__ == '__main__':
    import unittest
    unittest.main()
