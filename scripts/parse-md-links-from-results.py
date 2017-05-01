import argparse
from os import walk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir-name', dest='dir_name', action='store', required=True,
                        help='name of the directory that contains the results to be parsed')
    args = parser.parse_args()

    master_branch_results_path = "https://github.com/IGITUGraz/LTL/tree/master/"
    results_list = []
    for (dirpath, dirnames, filenames) in walk(args.dir_name):
        for ind, filename in enumerate(filenames):
            if filename == "result_details.md":
                results_list.append(dirpath + "/" + filename)

    wiki_text = ""
    for res in results_list:
        with open(res, 'r') as ifile:
            experiment_text = ifile.read()
            path = res.replace("result_details.md", "")
            experiment_text = experiment_text.replace('href="optimizee_parameters.yml"',
                                                      'href="' + master_branch_results_path + path + 'optimizee_parameters.yml"')
            experiment_text = experiment_text.replace('href="optimizer_parameters.yml"',
                                                      'href="' + master_branch_results_path + path + 'optimizer_parameters.yml"')
            experiment_text = experiment_text.replace('href="optima_coordinates.yml"',
                                                      'href="' + master_branch_results_path + path + 'optima_coordinates.yml"')
            wiki_text += experiment_text
            wiki_text += "\n\n *** \n"

    with open('results_for_wiki_parsed.md', 'w') as ofile:
        ofile.write(wiki_text)


if __name__ == '__main__':
    main()
