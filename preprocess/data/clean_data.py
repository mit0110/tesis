"""After running this script, all_data.pickle will contain a dictionary
where the keys are the names and the values are lists of types.
"""

files = [
    'book author.json',
    'books.json',
    'film actor.json',
    'film director.json',
    'locations.json',
    'musical group.json',
    'music group member.json',
    'tv actor.json',
    'tv program.json',
    'sparql']

from collections import defaultdict
import json
import pickle


def main():
    result = defaultdict(lambda : [])
    for filename in files:
        print "adding", filename
        f = open(filename)
        raw_json = f.read()
        f.close()
        d = json.loads(raw_json)['results']['bindings']
        for r in d:
            result[r['name']['value']].append(r['type_name']['value'])

        print " ... {} new named entities".format(len(d))

    result_file = open('all_data.pickle', 'w')
    pickle.dump(dict(result), result_file)
    result_file.close()


if __name__ == '__main__':
    main()