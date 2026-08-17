---
title: Performance Benchmark
weight: 20
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->


# Performance Benchmarking in FlagGems

It is recommended to follow the steps below to add test cases for a new operator.
These steps apply to Python-based operators as well as C++-wrapped operators.

{{% steps %}}

1. **Select the appropriate test file**

   <!--TODO(Qiming): remove the following constraints. -->
   Based on the type of operator, choose the corresponding file in the `benchmark`
   directory:

   - For reduction operators, add the test case to `test_reduction_perf.py`.

   - For tensor constructor operators, add the test case to `test_tensor_constructor_perf.py`.

   - If the operator doesn't fit into an existing category, you can add it to `test_special_perf.py`
     or create a new file for the new operator category.

1. **Check existing benchmark classes**

   Once you've identified the correct file, review the existing classes that inherit
   from the `Benchmark` structure to see if any fit the test scenario for your operator,
   specifically considering:

   - Whether the **metric collection** is suitable.

   - Whether the **input generation function** (`input_generator` or `input_fn`) is appropriate.

1. **Add test cases**

   Depending on the test scenario, follow one of the approaches below to add the test case:

   - **Using existing metric and input generator**

     If the existing metric collection and input generation function meet the requirements of your operator,
     you can add a line of `pytest.mark.parametrize` directly, following the code organization in the file.
     For example, see the operators in `test_binary_pointwise_perf.py`.

   - **Custom input generator**

     If the metric collection is suitable but the input generation function does not meet the operator's requirements,
     you can implement a custom `input_generator`.
     Refer to the `topk_input_fn` function in `test_special_perf.py` as an example of a custom input function
     for the `topk` operator.

   - **Custom metric and input generator**

     If neither the existing metric collection nor the input generation function meets the operator's needs,
     you can create a new class. The new class should define operator-specific metric collection logic
     and a custom input generator. You can refer to various `Benchmark` subclasses across the `benchmark` directory
     for examples.
{{% /steps %}}

## Enumerating and replaying benchmark cases

Benchmark families that expose their loop coordinates can list internal cases
without allocating tensors:

```shell
python -m pytest benchmark/test_addmm_.py --level core \
  --list-cases --output addmm-cases.json
```

The JSON report uses `flaggems.benchmark-case-list/v2`. It records one globally
unique `case_id` for every timing case, its dtype, the original shape coordinates,
and non-shape loop parameters. The opaque ID includes the pytest node identity and
the local loop coordinates. For `BlasBenchmark`, the metadata contains `b/m/n/k`
and `b_column_major`. Duplicate shapes remain separate cases because identity also
includes the original ordinal.

Run one or more cases by repeating the exact selector:

```shell
python -m pytest benchmark/test_addmm_.py --level core \
  --case-id 'benchmark/test_addmm_.py::test_addmm_::core::float16::0' \
  --record json --output addmm-result.json
```

Each migrated benchmark has two input stages: `get_case_iter()` plans ordered,
tensor-free cases, and `materialize_case()` constructs inputs only after a case
is selected. The selected and full benchmark paths consume the same planned
case sequence, and each emitted metric includes its `case_id`. Common BLAS,
generic, reduction, unary, and binary families provide this contract once for
their operators. A custom benchmark that replaces the family loop must provide
the same two stages; unsupported legacy benchmarks fail instead of reconstructing
cases approximately. The public operator ABI and default call arguments are not
duplicated in the case list.

External benchmark adapters can temporarily install a candidate implementation
before entering `use_gems()`:

```python
with flag_gems.testing.override_registered_op("addmm_", candidate):
    with flag_gems.use_gems(include=["addmm_"]):
        ...
```

The operator name must match exactly one public registry entry. The context
manager restores both registration tables, including when execution raises.
