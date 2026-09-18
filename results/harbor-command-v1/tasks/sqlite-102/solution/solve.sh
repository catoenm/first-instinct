#!/bin/sh
set -eu
python3 -c 'import sqlite3; c=sqlite3.connect('"'"'/app/data/orders.db'"'"'); c.execute("UPDATE orders SET status='"'"'ready'"'"' WHERE region=? AND status='"'"'pending'"'"'",('"'"'east'"'"',)); c.commit(); c.close()'
