
end = 260
deg = 180

in_distance = end - deg

if end<deg:
    out_distance = 360- deg +end
else:
    out_distance = 360- end +deg

print(in_distance)
print(out_distance)

if abs(in_distance) < abs(out_distance):
    if in_distance > 0:
        print(f"right by {abs(in_distance)}")
        pass
    else:
        print(f"left by {abs(in_distance)}")
        pass
else:
    if out_distance > 0:
        print(f"left by {abs(out_distance)}")
        pass
    else:
        print(f"right by {abs(out_distance)}")
        pass

if abs(in_distance)<180:
    print("left")
else:
    print("right")

for a in [True, False]:
    for b in [True, False]:
        result = a ^ b
        print(f"{a} XOR {b} = {result}")

x = -(360-366)
print(x)
print()

cíl = 320
start = 340

posun = 180 - cíl
cíl = 180


stupně = [0,90,180,359]

for i in stupně:
    print((i+posun)%360)


shift = 180 - end
end = 180
deg = (deg+shift)%360

in_distance = end - deg
print(f"")

end = 180
for i in range(0,359):
    in_distance = end - i
    print(f"{in_distance} from {i} to {end}")
