import json

with open("dog.json", "w") as f:
    with open("dog.txt") as rf:
        data = rf.readlines()

    data = list(map(lambda url: url.replace('\n', ''), data))
    json.dump(data, f, indent=4)

