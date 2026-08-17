# Copyright 2026 FlagOS Contributors
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

import pytest

import flag_gems


def _entry_for(operator):
    return next(item for item in flag_gems._FULL_CONFIG if item[0] == operator)


def test_override_registered_op_replaces_and_restores_registry():
    old_config = flag_gems._FULL_CONFIG
    old_by_func = flag_gems.FULL_CONFIG_BY_FUNC
    old_entry = _entry_for("addmm_")

    def candidate(*args, **kwargs):
        return old_entry[1](*args, **kwargs)

    with flag_gems.testing.override_registered_op("addmm_", candidate):
        new_entry = _entry_for("addmm_")
        assert new_entry[1] is candidate
        assert new_entry[2:] == old_entry[2:]
        assert any(
            new_entry in entries
            for entries in flag_gems.FULL_CONFIG_BY_FUNC.values()
        )

    assert flag_gems._FULL_CONFIG is old_config
    assert flag_gems.FULL_CONFIG_BY_FUNC is old_by_func
    assert _entry_for("addmm_") is old_entry


def test_override_registered_op_restores_after_error():
    old_config = flag_gems._FULL_CONFIG
    old_by_func = flag_gems.FULL_CONFIG_BY_FUNC

    with pytest.raises(RuntimeError, match="intentional"):
        with flag_gems.testing.override_registered_op("addmm_", lambda: None):
            raise RuntimeError("intentional")

    assert flag_gems._FULL_CONFIG is old_config
    assert flag_gems.FULL_CONFIG_BY_FUNC is old_by_func


def test_override_registered_op_requires_exact_public_key():
    with pytest.raises(ValueError, match="found 0"):
        with flag_gems.testing.override_registered_op(
            "missing.operator", lambda: None
        ):
            pass


def test_override_gems_op_is_independent_from_dispatcher_registry():
    old_config = flag_gems._FULL_CONFIG
    old_by_func = flag_gems.FULL_CONFIG_BY_FUNC
    default = lambda value: ("default", value)
    candidate = lambda value: ("candidate", value)

    assert flag_gems.testing.resolve_gems_op("unit_test_op", default) is default
    with flag_gems.testing.override_gems_op("unit_test_op", candidate):
        resolved = flag_gems.testing.resolve_gems_op("unit_test_op", default)
        assert resolved is candidate
        assert flag_gems.testing.gems_op_source("unit_test_op", resolved) == "override"
        assert flag_gems._FULL_CONFIG is old_config
        assert flag_gems.FULL_CONFIG_BY_FUNC is old_by_func

    assert flag_gems.testing.resolve_gems_op("unit_test_op", default) is default


def test_gems_op_case_is_scoped_and_restored():
    assert flag_gems.testing.current_gems_op_case("addmm_") is None
    case_id = "benchmark/test_addmm_.py::test_addmm_::core::float16::0"
    with flag_gems.testing.gems_op_case("addmm_", case_id):
        assert (
            flag_gems.testing.current_gems_op_case("addmm_")
            == case_id
        )
        assert flag_gems.testing.current_gems_op_case("abs") is None
    assert flag_gems.testing.current_gems_op_case("addmm_") is None
