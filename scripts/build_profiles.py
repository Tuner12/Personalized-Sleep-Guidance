#!/usr/bin/env python3
"""Profile Agent: write and run code that builds the profile packet P for each case (Section 3.1).

    python scripts/build_profiles.py --dataset <path> --notes <field documentation> --out outputs/profiles
"""

from sleepguide.stage1.profile_agent import main

if __name__ == "__main__":
    main()
