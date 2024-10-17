import json
import pandas as pd


def convert_kudos(n_kudos):
    bins = [0, 20, 50, 200, 1000, 10000]
    labels = ["not popular", "slightly popular", "moderately popular", "popular", "very popular"]

    category = pd.cut([int(n_kudos)], bins=bins, labels=labels)
    category_num = pd.Categorical(category, categories=labels)

    return category, int(category_num.codes[0])

def main():

    with open("sparql_data.json") as f:
        data = json.load(f)

    # to get info of each line
    #print(data["results"]["bindings"][0]["kudos"])

    for entry in data["results"]["bindings"]:
        n_kudos = entry["kudos"]["value"]

        entry["kudos_label"], entry["kudos_categorical"] = convert_kudos(n_kudos)




if __name__ == "__main__":
    main()