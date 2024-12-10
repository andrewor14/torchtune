# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

./run_normal_qat.sh
USE_QAT=true ./run_it.sh
USE_QAT=true ./eval_it.sh
USE_QAT=false ./run_it.sh
USE_QAT=false ./eval_it.sh
