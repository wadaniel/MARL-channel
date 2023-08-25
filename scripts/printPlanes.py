from helpers import fieldToState, calcControl, distribute_field, getHeights

if __name__ == "__main__":

    lower = 0.
    upper = 50
    heights = getHeights(lower, upper)
    print(heights)
