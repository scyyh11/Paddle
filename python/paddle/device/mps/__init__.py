# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from typing_extensions import TypeAlias

from paddle.base import core

if TYPE_CHECKING:
    from paddle import MPSPlace

    _MPSPlaceLike: TypeAlias = Union[
        MPSPlace,
        str,  # some string like "mps:0", etc.
        int,  # some int like 0, 1, etc.
    ]


def device_count() -> int:
    '''
    Return the number of MPS devices available.

    Returns:
        int: the number of MPS devices available.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.mps.device_count()
    '''

    num_mps = (
        core.get_mps_device_count()
        if hasattr(core, 'get_mps_device_count')
        else 0
    )

    return num_mps


def synchronize(device: _MPSPlaceLike | None = None) -> None:
    '''
    Wait for all operations in the MPS stream to complete.

    Args:
        device (MPSPlaceLike | None, optional): The MPS device to synchronize.
            If None, synchronizes the current device. Defaults to None.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> paddle.device.mps.synchronize()
    '''
    # MPS uses unified memory, so synchronization is typically not needed
    # But we provide this for API compatibility
    pass

