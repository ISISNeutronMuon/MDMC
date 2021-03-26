import re
from urllib.parse import quote

with open('/mdmc/.env', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        if re.search('^UNAME=', line):
            UNAME = quote(line[6:-1], safe='')
        elif re.search('^PWORD=', line):
            PWORD = quote(line[6:-1], safe='')

try:
    print(UNAME+':'+PWORD)
except NameError as error:
    raise NameError('Both `UNAME` and `PWORD` must be provided in `.env` file'
                    ) from error
