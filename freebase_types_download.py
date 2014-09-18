"""
IEPY's Freebase type instance names downloader (to be used with the Literal NER).

Usage:
    download_freebase_type.py <freebase_type_name> <output_file> [<cursor>] [options]
    download_freebase_type.py -h | --help | --version

Options:
  -h --help             Show this screen
  --version             Version number
  --aliases             Include instance aliases
  --to-lower            Convert non acronyms to lowercase
"""

import codecs
import json
import pickle
import urllib

from docopt import docopt

def download_freebase_type(type_name, dest_filename, normalizer=None,
                           aliases=False, last_cursor=None):
    if not normalizer:
        normalizer = lambda x: x

    # https://developers.google.com/freebase/v1/mql-overview
    service_url = 'https://www.googleapis.com/freebase/v1/mqlread'
    query = [{'type': type_name, 'name': None, '/type/object/type': []}]
    # if aliases:
    #     query[0]['/common/topic/alias'] = []

    recovered_ne = []
    if not last_cursor:
        params = {'query': json.dumps(query)}
        url = service_url + '?' + urllib.urlencode(params) + '&cursor'
        response = json.loads(urllib.urlopen(url).read())
        cursor = response['cursor']

        for result in response['result']:
            name = normalizer(result['name'])
            types = result['/type/object/type']
            # if aliases:
            #     for name in result['/common/topic/alias']:
            #         name = normalizer(name)
            recovered_ne.append((name, types))

    else:
        cursor = unicode(last_cursor)

    while cursor:
        print "Loading {} instances".format(len(recovered_ne))
        params = {'query': json.dumps(query), 'cursor': cursor}
        url = service_url + '?' + urllib.urlencode(params)
        response = json.loads(urllib.urlopen(url).read())
        if not 'cursor' in response:
            print "No cursor found, start again!", cursor
            break
        cursor = response['cursor']

        for result in response['result']:
            if result['name']:
                name = normalizer(result['name'])
                types = result['/type/object/type']
            # if aliases:
            #     for name in result['/common/topic/alias']:
            #         name = normalizer(name)
                recovered_ne.append((name, types))

    f = open(dest_filename, 'w')
    pickle.dump(recovered_ne, f)
    f.close()


def to_lower_normalizer(name):
    """Utility normalizer that converts a name to lowercase unless it is an
    acronym. To be used as parameter of download_freebase_type().
    """
    words = name.split()
    result = []
    for w in words:
        if not w.isupper():
            w = w.lower()
        result.append(w)
    return ' '.join(result)


if __name__ == '__main__':
    opts = docopt(__doc__, version=0.1)
    freebase_type_name = opts['<freebase_type_name>']
    output_file = opts['<output_file>']
    cursor = opts['<cursor>'] if '<cursor>' in opts else None
    aliases = opts['--aliases']
    to_lower = opts['--to-lower']

    if to_lower:
        normalizer = to_lower_normalizer
    else:
        normalizer = None
    download_freebase_type(freebase_type_name, output_file, normalizer,
                           aliases, cursor)