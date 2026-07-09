"""
Take in a file path of the final scene and return an calculated final product.

Detailed pipeline:
1. crop image
2. filter image with morphical edge enhancement
3. calculate where stitches (aka most balck points per column) are
4. plot the final products

Example:
    - will write after creating main
"""

from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_image(file: str, name: str) -> None:
    """Crop an image and save it to a fixed directory.
    args:
    - file : a string of a path to an image
    - name : id of student to name the file
    """
    img = Image.open(file)
    res = img.crop((827, 472, 1132, 582))
    # res = img.crop((a, b, c, d))
    # extracts from x=a-c, y=b-d
    res.save(f"/home/sov1/OpenSutureNet/final_frame/{name}.png")
    print(f"Saved to /home/sov1/OpenSutureNet/final_frame/{name}.png")

def process_scene(file: str) -> list[list[int]]:
    """
    Opening performs erosion followed by dialtion on a binary image.
    It is commonly used to remove samll noise and unwanted foreground objects
    while preserving the overall shape of larger objects.

    Implementation:
    - create 3*3 kenel to define the nerighborhood used for the morphological operation.
    - apply cv2.morph_open to perfrom erosion followed by dilating,
    removing small foreground noise while perserving larger objects.

    Source: https://www.geeksforgeeks.org/python/python-opencv-morphological-operations/

    args:
    - file : a string of file path to a croped scene
    returns:
    - opened: an array of final processed balck and white frame
    """
    img = cv2.imread(file, 0)
    bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    k = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(bin, cv2.MORPH_OPEN, k)
    return opened

def _count_black_per_column(processed_scene: list[list[int]]) -> list[int]:
    """Return amount of black point per columns from croped and filtered scene
    args:
    - processed_scene : an array result of process_scene funciton
    returns:
    - black_per_column : an array of number of black points per column
    """
    black_per_column = []
    for x in range(0, 305):
        count_black = 0
        for y in range(0, 110):
            if processed_scene[y][x] == 0:
                count_black += 1
        black_per_column.append(count_black)
    return black_per_column

def _get_med_outliers(black_per_column: list[int]):
    """Return a median of higher outlier data to set as how many points we should get.
    Modified from Source - https://stackoverflow.com/q/11686720
    Posted by aaren
    Retrieved 2026-07-06, License - CC BY-SA 3.0
    args:
    - black_per_column : an array of number of black points per column
    returns:
    - output : interger of median of all the outliers
    """
    u = np.mean(black_per_column)
    s = np.std(black_per_column)
    filtered = [e for e in black_per_column if e > u + 1.5 * s]
    return int(np.median(filtered))

def get_black_point_location(processed_scene: list[list[int]]) -> list[int]:
    """Return a list of column index where black points are most aggregated.
    args:
    - processed_scene : an array result of process_scene funciton
    returns:
    - final_column_index : an array of column index
    """
    black_per_column = _count_black_per_column(processed_scene)
    final_column_index = []
    for i in range(0, 305):
        if black_per_column[i] > _get_med_outliers(black_per_column):
            final_column_index.append(i)
    print(final_column_index)

def _group_close_points(input: list[int]) -> list[list[int]]:
    """Return a list of list of close points.
    args:
    - input: [46, 47, 48, 68, 69, 86, 87, 88, 89, 110, 111, 112]
    returns:
    - output: [[46, 47, 48], [68, 69], [86, 87, 88, 89], [110, 111, 112]]
    """
    length = len(input)
    output = [[input[0]]]
    count = 0
    for i in range(1, length):
        if input[i] - input[i-1] == 1:
            output[count].append(input[i])
        else:
            new = [input[i]]
            count += 1
            output.append(new)
    return output

def _group_min_max(input: list[list[int]]) -> list[list[int]]:
    """Return a list of list of min and max in close points.
    args:
    - input: [[46, 47, 48], [68, 69], [86, 87, 88, 89], [110, 111, 112]]
    returns:
    - output: [[46, 48], [68, 69], [86, 89], [110, 112]]
    """
    length = len(input)
    output = []
    for i in range(0, length):
        min = min(input[i])
        max = max(input[i])
        output.append([min, max])
    return output

def _group_start_end(input: list[list[int]]) -> list[list[int]]:
    """Return a list of list of start and end in close points.
    args:
    - input: [[46, 47, 48], [68, 69], [86, 87, 88, 89], [110, 111, 112]]
    returns:
    - output: [[48, 68], [69, 86], [89, 110]]
    """
    length = len(input)
    output = []
    for i in range(0, length - 1):
        start = input[i][len(input[i]) - 1]
        end = input[i + 1][0]
        output.append([start, end])
    return output

def _calculate_distance(input: list[list[int]]) -> list[int]:
    """Return a list of distance start and end.
    args:
    - input: [[48, 68], [69, 86], [89, 110]]
    returns:
    - output: [20, 17, 21]
    """
    length = len(input)
    output = []
    for i in range(0, length):
        output.append(input[i][1] - input[i][0])
    return output

def plot_final_product(processed_scene: list[list[int]], black_points: list[int], id: str) -> None:
    """Return a final plot of the final stitch quality assessment
    args:
    - processed_scene : an array of black and white final stitches (x = 305, y = 110)
    - black_points : an array of where most black points are accumulated
    - id : a string of student ID to be reported
    """
    grouped = _group_close_points(black_points)
    grouped_min_max = _group_min_max(grouped)
    grouped_start_end = _group_start_end(grouped)
    distance = _calculate_distance(grouped_start_end)
    # plot title
    plt.title(f"{id} : Quality of Final Product")
    # plot the final processed product
    plt.imshow(processed_scene, cmap='gray')
    plt.axis('image')
    # plot vertical red line of stitch location
    for x in black_points:
        for y in range(0, 110):
            plt.plot(x, y, marker='o', markersize=2, color='red', alpha=0.05)
    # plot distance blue line
    for item in grouped_start_end:
        for x in range(item[0] + 1, item[1]):
            plt.plot(x, 80, marker='o', markersize=1, color='blue')
    # plot described distance
    count = len(grouped)
    for i in range(0, count - 1):
        plot_point = grouped_start_end[i][0]
        plot_message = str(distance[i])
        plt.text(plot_point + 1, 90, plot_message, fontsize=10, color="blue")
    # plot count of stitches
    count_message = f"count = {len(grouped_min_max)}"
    plt.text(200, 20, count_message, fontsize=15, color="blue", bbox=dict(facecolor="yellow", edgecolor="green", linewidth=2, alpha=0.5))
    plt.grid(alpha = 0.2)
    plt.show()