import json
import pandas as pd
import csv
from datetime import datetime


def convert_kudos(n_kudos):
    bins = [0, 20, 50, 200, 1000, 10000]
    labels = ["not popular", "slightly popular", "moderately popular", "popular", "very popular"]

    category = pd.cut([int(n_kudos)], bins=bins, labels=labels)
    category_num = pd.Categorical(category, categories=labels)

    return category[0], int(category_num.codes[0])

def main():

    # with open("sparql_data_new.json") as f:
    #     data = json.load(f)

    with open("sparql_data_csv.csv") as f:
        reader = csv.DictReader(f)
        data_list = list()
        for row in reader:
            data_list.append(row)

    # to get info of each line
    #print(data["results"]["bindings"][0]["kudos"])

    data_new = dict()

    for entry in data_list:

        time = "%Y-%m-%dT%H:%M:%S"
        # collecting necessary for preprocessing values
        n_kudos = entry["kudos"]
        title = entry["title"]
        pack_date = datetime.strptime(entry["packDate"], time).date()
        status = entry["pubStat"]

        data_new[title] = data_new.get(title, entry)

        # update the data
        if data_new[title]["pubStat"] != status and status == "Completed":
            data_new[title] = entry
        elif datetime.strptime(data_new[title]["packDate"], time).date() < pack_date:
            data_new[title] = entry
        elif int(data_new[title]["kudos"]) < int(n_kudos):
            data_new[title] = entry

        data_new[title]["kudos_label"], data_new[title]["kudos_categorical"] = convert_kudos(n_kudos)

        # add day difference between modified date and packaged date
        mod_date = datetime.strptime(data_new[title]["modDate"], "%Y-%m-%d").date()
        pack_date = datetime.strptime(data_new[title]["packDate"], time).date()
        data_new[title]["day_diff"] = (pack_date - mod_date).days
    
    with open("sparql_data_preprocess.json", "w") as f:
        json.dump(data_new, f, indent=1)



if __name__ == "__main__":
    main()