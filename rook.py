start = [0, 0]
end = [7, 7]
taken = [[1, 0], [1, 1], [1, 2], [1, 3]]
queue = []
queue.append([start[0], start[1], -1])
visited = []
maze = []
for i in range(8):
    maze.append([".", ".", ".", ".", ".", ".", ".", "."])
    visited.append([0, 0, 0, 0, 0, 0, 0, 0])
maze[start[0]][start[1]] = "S"
maze[end[0]][end[1]] = "E"
for i in taken:
    maze[i[0]][i[1]] = "X"
while len(queue) > 0:
    point = queue.pop(0)
    if end[0] == point[0] and end[1] == point[1]:
        print(point[2] + 1)
        break
    current = point[2] + 1
    if point not in taken and visited[point[0]][point[1]] == 0:
        visited[point[0]][point[1]] = current
        for i in range(point[0], -1, -1):
            if [i, point[1]] in taken:
                break
            if visited[i][point[1]] == 0:
                queue.append([i, point[1], current])
        for i in range(point[0], 8):
            if [i, point[1]] in taken:
                break
            if visited[i][point[1]] == 0:
                queue.append([i, point[1], current])
        for i in range(point[1], -1, -1):
            if [point[0], i] in taken:
                break
            if visited[point[0]][i] == 0:
                queue.append([point[0], i, current])
        for i in range(point[1], 8):
            if [point[0], i] in taken:
                break
            if visited[point[0]][i] == 0:
                queue.append([point[0], i, current])

for i in maze:
    for j in i:
        print(j, end="   ")
    print()
