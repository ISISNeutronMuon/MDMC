import re
from urllib.parse import quote

with open('/mdmc/.env', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        if re.search('^UNAME=', line):
            UNAME = quote(line[6:-1], safe='')
        elif re.search('^TOKEN=', line):
            TOKEN = quote(line[6:-1], safe='')

try:
    print(UNAME+':'+TOKEN)
except NameError as error:
    raise NameError('Both `UNAME` and `TOKEN` must be provided in `.env` file'
                    ) from error
