#!/bin/bash

rsync -rav -e ssh --include="*/" --include='*.py' --include='*.bash'  --exclude="*" --prune-empty-dirs . turandot:TurbulentWater
