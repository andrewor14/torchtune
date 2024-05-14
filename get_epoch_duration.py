from datetime import datetime

import re
import sys

args = sys.argv
if len(args) != 2:
    print("Usage: python get_epoch_duration.py [/path/to/run.log]")
    sys.exit(1)

print("Parsing log ", args[1])
with open(args[1], "r") as f:
    fmt = "%Y-%m-%d:%H:%M:%S"
    lines = f.readlines()
    epoch = 0
    all_durations = []
    for i, line in enumerate(lines):
        if "Model checkpoint of size" in line:
            # ... 2024-04-28:17:03:36,395 INFO     [_checkpointer.py:582] Model checkpoint of size
            epoch_end = re.match(".*(2024.*),.* INFO", line).groups()[0]
            epoch_end = datetime.strptime(epoch_end, fmt)
            # Find last line with timestamp
            j = i - 1
            while j >= 0:
                match = re.match(".*(2024.*),.* INFO", lines[j])
                if match is not None:
                    break
                j -= 1
            if j < 0:
                raise ValueError("Could not find previous line?")
            epoch_start = datetime.strptime(match.groups()[0], fmt)
            duration = (epoch_end - epoch_start).seconds / 3600
            print("Epoch %s took %0.3f hours" % (epoch, duration))
            epoch += 1
            all_durations.append(duration)

avg_duration = sum(all_durations) / len(all_durations)
print("Average epoch duration: %0.3f hours" % avg_duration)
