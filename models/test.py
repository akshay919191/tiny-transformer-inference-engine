import math , argparse

parser = argparse.ArgumentParser(description = "area")

parser.add_argument("--radius" , default = 1 , nargs = '?' , type = int , help = "radius of cylinder")
parser.add_argument("--height" , default = 2 , nargs = '?' , type = int , help = "height of cylinder")


args = parser.parse_args()
def area(radius , height):
    return math.pi * radius ** 2 * height

print(area(args.radius , args.height))