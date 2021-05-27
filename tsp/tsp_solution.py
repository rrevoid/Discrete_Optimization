import numpy as np
from tqdm import tqdm
import plotly.graph_objs as go


def process_input_data(input_data):
    lines = input_data.split('\n')
    count = int(lines[0])
    points = np.zeros(shape = (count, 2))
    for i in range(1, count + 1):
        coords = np.array(list(map(float, lines[i].split())))
        points[i - 1] = coords
    return points, count


class TSPSolver():
    def __init__(self, input_data):
        # Инициализируем поля класса
        self.points, self.count = process_input_data(input_data)
        self.distances = self.get_distances()
        self.order = self.get_best_greedy_solution()
        self.paths = np.array([self.order.copy()])

    def get_distances(self):
        # Вычисление матрицы расстояний между городами
        distances = np.zeros(shape=(2, self.count, self.count, 2))
        distances[0, :] = self.points
        distances[1, :] = self.points.reshape(self.count, 1, 2)
        distances = np.sqrt(np.sum((distances[0, :] - distances[1, :]) ** 2, axis=-1))
        return distances

    def count_path_len(self, order):
        # Вычисление длины маршрута
        obj = self.distances[order[-1], order[0]]
        for index in range(self.count - 1):
            obj += self.distances[order[index], order[index + 1]]
        return obj

    def get_greedy_solution(self, start_vertex):
        # Получение жадного решения из заданной вершины
        order = [start_vertex]
        available = {i for i in range(self.count)}
        available.remove(order[0])
        for i in range(self.count - 1):
            min_dist = float("+inf")
            min_vertex = -1
            for vertex in available:
                dist = self.distances[order[i], vertex]
                if dist < min_dist:
                    min_dist = dist
                    min_vertex = vertex
            if min_vertex != -1:
                order.append(min_vertex)
                available.remove(min_vertex)

        for vertex in available:
            order.append(vertex)
        return order

    def get_best_greedy_solution(self):
        # Получение первоначального решения путем выбора
        # из нескольких жадных решений
        number = int(np.log2(self.count))
        indexes = np.random.choice(self.count, number, replace=False)
        min_order = self.get_greedy_solution(indexes[0])
        min_dist = self.count_path_len(min_order)
        for i in tqdm(range(1, number)):
            order = self.get_greedy_solution(indexes[i])
            dist = self.count_path_len(min_order)
            if dist < min_dist:
                min_order = order
        return min_order

    def invert(self, start, end):
        # Разворот части пути
        if start > end:
            end, start = start - 1, end + 1
        rev = reversed(self.order[start: end + 1])
        self.order[start: end + 1] = rev

    def k_opt(self, t1):
        # Реализация k-opt для вершины t1

        changed = False

        # сохраняем начальный порядок
        min_order = self.order.copy()
        t2 = (t1 + 1) % self.count

        # доступные для рассмотрения вершины
        available = {i for i in range(self.count)}
        available.remove(self.order[t1])
        available.remove(self.order[t2])
        available.remove(self.order[(t2 + 1) % self.count])

        # Разница между путями
        difference = 0
        min_diff = 0
        k_max = int(4 * np.sqrt(self.count))
        for k in range(k_max):
            d1 = d2 = self.distances[self.order[t1], self.order[t2]]
            t3 = -1
            # Ищем ребро (t2, t3) < (t1, t2)
            for ind in range(self.count):
                if self.order[ind] in available and ind != t2:
                    d = self.distances[self.order[ind],self. order[t2]]
                    if d < d2:
                        d2 = d
                        t3 = ind
            if t3 == -1:
                break
            available.remove(self.order[t3])
            t4 = (t3 - 1) % self.count

            # Считаем изменение длины пути
            difference += d2 + self.distances[self.order[t1], self.order[t4]] - \
                          d1 - self.distances[self.order[t3], self.order[t4]]
            self.invert(t2, t4)

            # Обновляем лучшее решение
            if difference < min_diff:
                changed = True
                min_diff = difference
                min_order = self.order.copy()
        return changed, min_order

    def solve(self):
        found_solution = False
        max_patience = -1
        max_orders = int(3 * np.log2(self.count))
        prev_orders = []
        pbar = tqdm(total=self.count)
        while not found_solution:
            length = len(prev_orders)
            if length > 0:
                rand = np.random.randint(0, 20)
                if rand == 0:
                    rand_ind = np.random.randint(0, length)
                    self.order = prev_orders[rand_ind]
            patience = 0
            permutation = np.random.permutation(self.count)
            # Запускаем k-opt для всех вершин
            for i in permutation:
                if length == max_orders:
                    rand_ind = np.random.choice(self.count, 1, p=[max_orders - i for i in range(max_orders)])
                    prev_orders[rand_ind] = self.order.copy()
                else:
                    prev_orders.append(self.order.copy())
                changed, self.order = self.k_opt(self.order.index(i))
                if not changed:
                    patience += 1
                else:
                    self.paths = np.vstack((self.paths, self.order.copy()))
            if patience > max_patience:
                pbar.update(patience - max_patience)
                max_patience = patience

            # Если за обход всех вершин ни разу не обновили, то завершаем поиск решения
            if patience == self.count:
                found_solution = True
        pbar.close()
        self.animate()
        solution = f"Длина: {self.count_path_len(self.order)}\nПорядок вершин: {self.order}"
        return solution

    def animate(self):
        # Анимация нахождения решения
        paths = np.hstack((self.paths, self.paths[:,0].reshape(-1,1)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.points[paths[0], 0], y=self.points[paths[0], 1],
                                 mode='lines+markers', name='Traveling salesman path'))

        frames = []
        frame_number = paths.shape[0]
        for i in range(1, frame_number):
            frames.append(go.Frame(data=[go.Scatter(x=self.points[paths[i], 0], y=self.points[paths[i], 1])]))

        fig.frames = frames

        fig.update_layout(legend_orientation="h",
                          legend=dict(x=.5, xanchor="center"),
                          updatemenus=[dict(type="buttons", buttons=[
                              dict(label="►", method="animate", args=[None, {"fromcurrent": True}]),
                              dict(label="❚❚", method="animate",
                                   args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                  "mode": "immediate",
                                                  "transition": {"duration": 0}}])])],
                          margin=dict(l=0, r=0, t=0, b=0))
        fig.update_traces(hoverinfo="all", hovertemplate="x: %{x}<br>y: %{y}")

        fig.show()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        solver = TSPSolver(input_data)
        print(solver.solve())
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')