"""
Take the given csv file of the dataset to see the
characteristic of distribution of the dataset include the following:

1. Score distribution per OSATS dimension
2. PRE and POST Split
3. Inter rater agreement
4. Spearman correlation between all pair of dimension
5. How SUTURES is associated to GRS

This will be major consideration for splitting training and testing
datset.
"""
import csv
import matplotlib.pyplot as plt
import numpy as np
import krippendorff
from scipy.stats import spearmanr
import seaborn as sns

def _read_based_on_evaluater(evaluater: str, file: str) -> list[list[str]]:
    """Return a list of list of score from a given
    evaluater ('A', 'B', 'C') and csv file."""
    raw_score = []
    with open(file, "r") as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            if row[4] == evaluater:
                raw_score.append(row)
    return raw_score

def _score_distribution_per_osats(raw_score: list[list[str]]) -> list[list[int]]:
    """Return a list of all counts of score distribution per score."""
    count_all = [[0, 0, 0, 0, 0] for _ in range(0,8)]
    for i in range(0, 8):
        for item in raw_score:
            if item[6 + i] == '1':
                count_all[i][0] += 1
            elif item[6 + i] == '2':
                count_all[i][1] += 1
            elif item[6 + i] == '3':
                count_all[i][2] += 1
            elif item[6 + i] == '4':
                count_all[i][3] += 1
            else:
                count_all[i][4] += 1
    return count_all

def _plot_score_distribution(score_distribution: list[list[int]], title: str) -> None:
    """Print a firgure of 8 bar graphs for distribution for 8 OSTAS"""
    names = ["Respect", "Motion", "Instrument", "Suture",
             "Flow", "Knowledge", "Performance", "Final Quality"]
    fig, axes = plt.subplots(nrows=2, ncols=4,
                             figsize=(8, 4), layout="constrained")
    fig.suptitle(f"OSATS Score Distribution - Evaluator {title}")

    for x in range(2):
        for y in range(4):
            i = x * 4 + y
            categories = [1, 2, 3, 4, 5]
            values = score_distribution[i]
            axes[x, y].bar(categories, values)
            axes[x, y].set_title(names[i])
            axes[x, y].set_ylim(0, 150)
            axes[x, y].set_xlabel('Score')
            axes[x, y].set_ylabel('No of occurance')
    plt.show()

def plot_score_distribution_per_evaluater(file: str) -> None:
    """Read csv file, filter only an evaluater ('A', 'B', 'C'), then plot it"""
    # Plot data distribution for evaluater A
    raw_score = _read_based_on_evaluater('A', file)
    score_distribution = _score_distribution_per_osats(raw_score)
    _plot_score_distribution(score_distribution, 'A')

    # Plot data distribution for evaluater B
    raw_score = _read_based_on_evaluater('B', file)
    score_distribution = _score_distribution_per_osats(raw_score)
    _plot_score_distribution(score_distribution, 'B')

    # Plot data distribution for evaluater C
    raw_score = _read_based_on_evaluater('C', file)
    score_distribution = _score_distribution_per_osats(raw_score)
    _plot_score_distribution(score_distribution, 'C')


def _read_pre(file: str) -> list[list[str]]:
    """Return a list of list of score from a given
    of only PRE tag"""
    raw_score = []
    with open(file, "r") as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            if row[2] == 'PRE':
                raw_score.append(row)
    return raw_score

def _read_post(file: str) -> list[list[str]]:
    """Return a list of list of score from a given
    of only POST tag"""
    raw_score = []
    with open(file, "r") as f:
        data = csv.reader(f)
        next(data)
        for row in data:
            if row[2] == 'POST':
                raw_score.append(row)
    return raw_score

def _accumulate_grs(raw_score: list[list[str]]) -> list[int]:
    """Return a list of integers of grs form a raw_score"""
    accumulated_grs = []
    for item in raw_score:
        accumulated_grs.append(int(item[-1]))
    return accumulated_grs

def pre_post_stats_grs(file: str) -> dict:
    """Return a dictionary of mean grs"""
    output = {'PRE': {'mean': 0, 'median': 0},
                'POST': {'mean': 0, 'median': 0}}
    pre_raw_score = _read_pre(file)
    post_raw_score = _read_post(file)

    output['PRE']['mean'] = np.mean(_accumulate_grs(pre_raw_score))
    output['POST']['mean'] = np.mean(_accumulate_grs(post_raw_score))

    output['PRE']['median'] = np.median(_accumulate_grs(pre_raw_score))
    output['POST']['median'] = np.median(_accumulate_grs(post_raw_score))
    return output


def _read_grs_and_rater(file: str) -> list[list]:
    """Return a list of list representing matrix for krippendorff.alpha
    function: column = video, row = evaluater
    row 0 -- A
    row 1 -- B
    row 2 -- C
    """
    grs_and_rater_data = [[] for _ in range(0, 3)]
    with open(file, "r") as f:
        data = csv.reader(f)
        next(data)
        for item in data:
            if item[4] == 'A':
                grs_and_rater_data[0].append(int(item[-1]))
            elif item[4] == 'B':
                grs_and_rater_data[1].append(int(item[-1]))
            elif item[4] == 'C':
                grs_and_rater_data[2].append(int(item[-1]))
    return grs_and_rater_data

def calculate_krippendorff_alpha(file: str) -> None:
    """Report Krippendorff's alpha value for reliability of
    3 evaluaters from the input csv file"""
    grs_data = _read_grs_and_rater(file)
    grs_AB = [grs_data[0], grs_data[1]]
    grs_AC = [grs_data[0], grs_data[2]]
    grs_BC = [grs_data[1], grs_data[2]]

    # Calculate Krippendorff's Alpha -- for non-nominal data
    alpha_value = krippendorff.alpha(
        reliability_data=grs_data,
        level_of_measurement="ordinal"
    )
    alpha_AB = krippendorff.alpha(reliability_data=grs_AB,
                                  level_of_measurement="ordinal")
    alpha_AC = krippendorff.alpha(reliability_data=grs_AC,
                                  level_of_measurement="ordinal")
    alpha_BC = krippendorff.alpha(reliability_data=grs_BC,
                                  level_of_measurement="ordinal")

    print("-------------------------------------------")
    print(f"Krippendorff's alpha = {alpha_value}")
    print("-------------------------------------------")

    print(f"Krippendorff's alpha A-B= {alpha_AB}")
    print(f"Krippendorff's alpha A-C= {alpha_AC}")
    print(f"Krippendorff's alpha B-C= {alpha_BC}")

    print("-------------------------------------------")
    print("Overall Interpretation:")
    if alpha_value == 1.0:
        print("Perfect agreement")
    elif alpha_value >= 0.8:
        print("Strong agreement (generally acceptable)")
    elif alpha_value >= 0.67:
        print("Tentative conclusions may be acceptable")
    elif alpha_value >= 0.0:
        print("Poor agreement")
    elif alpha_value == 0.0:
        print("Agreement is no better than chance")
    else:
        print("Worse than chance")

    print("-------------------------------------------")

def _read_row_of_vid(file: str) -> list[list[int]]:
    """Return a list of score in the given file,
    which a row is videos and column is osats"""
    row_of_vid_data = []
    with open(file, "r") as f:
        data = csv.reader(f)
        next(data)
        for item in data:
            row_of_vid_data.append([int(item[6 + i]) for i in range(0, 8)])
    return row_of_vid_data

def _calculate_correlation(row_of_vid_data: list[list[int]]) -> tuple:
    """Return a tuple of matrix of correlation values
    and matrix of p_values"""
    row_of_vid_data = _read_row_of_vid(file)
    correlation_matrix, p_values = spearmanr(row_of_vid_data)
    return correlation_matrix, p_values

def visualize_correlation_pvals(file:str) -> None:
    """Print out a correlation and pvalues heatmap"""
    osats_labels = ["Respect", "Motion", "Instrument", "Suture", "Flow",
                    "Knowledge", "Performance", "Final Quality"]

    row_of_vid_data = _read_row_of_vid(file)
    correlation_matrix, p_values = _calculate_correlation(row_of_vid_data)

    # Correlation plot
    plt.figure(figsize=(8, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,  # Display the numbers inside each cell
        fmt=".2f",  # Format numbers to 2 decimal places
        cmap="coolwarm",  # Choose a color palette (e.g., 'viridis', 'YlGnBu')
        linewidths=0.5,  # Add a thin border between cells
        square=True,  # Force cells to be perfect squares
        xticklabels=osats_labels,
        yticklabels=osats_labels
    )
    plt.title("P_values for each OSATS rubric")
    plt.show()