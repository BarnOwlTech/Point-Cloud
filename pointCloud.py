from math import cos, sin, radians, sqrt, acos, atan, asin
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, TextBox
import sys


class Matrix4f():
    """Класс для работы с матрицами 4x4 (однородные координаты)"""

    def __init__(self, matrix=None, fill=None, diag_fill=None):
        if matrix is not None:
            self.matrix = np.array(matrix, dtype=float)
        elif fill is not None:
            self.matrix = np.full((4, 4), fill, dtype=float)
        elif diag_fill is not None:
            self.matrix = np.eye(4, dtype=float) * diag_fill
        else:
            self.matrix = np.eye(4, dtype=float)

    def __str__(self):
        m = self.matrix
        res = '\n'
        for i in range(4):
            res += ' | '.join(f'{elem:>8.3f}' for elem in m[i]) + '\n'
        return res

    def __mul__(self, other):
        if isinstance(other, Matrix4f):
            return Matrix4f(matrix=self.matrix @ other.matrix)
        elif isinstance(other, np.ndarray) and other.shape == (4,):
            # Умножение матрицы на вектор
            return self.matrix @ other
        return NotImplemented

    @staticmethod
    def translation(tx, ty, tz):
        """Матрица трансляции"""
        m = np.eye(4)
        m[0, 3] = tx
        m[1, 3] = ty
        m[2, 3] = tz
        return Matrix4f(matrix=m)

    @staticmethod
    def rotation_x(angle):
        """Вращение вокруг оси X"""
        m = np.eye(4)
        m[1, 1] = cos(angle)
        m[1, 2] = -sin(angle)
        m[2, 1] = sin(angle)
        m[2, 2] = cos(angle)
        return Matrix4f(matrix=m)

    @staticmethod
    def rotation_y(angle):
        """Вращение вокруг оси Y"""
        m = np.eye(4)
        m[0, 0] = cos(angle)
        m[0, 2] = sin(angle)
        m[2, 0] = -sin(angle)
        m[2, 2] = cos(angle)
        return Matrix4f(matrix=m)

    @staticmethod
    def rotation_z(angle):
        """Вращение вокруг оси Z"""
        m = np.eye(4)
        m[0, 0] = cos(angle)
        m[0, 1] = -sin(angle)
        m[1, 0] = sin(angle)
        m[1, 1] = cos(angle)
        return Matrix4f(matrix=m)

    @staticmethod
    def rotation_axis(angle, vx, vy, vz):
        """Вращение вокруг произвольной оси"""
        # Нормализуем вектор
        length = sqrt(vx * vx + vy * vy + vz * vz)
        vx, vy, vz = vx / length, vy / length, vz / length

        m = np.eye(4)
        cos_a = cos(angle)
        sin_a = sin(angle)
        one_minus_cos = 1 - cos_a

        m[0, 0] = cos_a + vx * vx * one_minus_cos
        m[0, 1] = vx * vy * one_minus_cos - vz * sin_a
        m[0, 2] = vx * vz * one_minus_cos + vy * sin_a

        m[1, 0] = vx * vy * one_minus_cos + vz * sin_a
        m[1, 1] = cos_a + vy * vy * one_minus_cos
        m[1, 2] = vy * vz * one_minus_cos - vx * sin_a

        m[2, 0] = vx * vz * one_minus_cos - vy * sin_a
        m[2, 1] = vy * vz * one_minus_cos + vx * sin_a
        m[2, 2] = cos_a + vz * vz * one_minus_cos

        return Matrix4f(matrix=m)


class Bunny3D:
    """Класс для работы и визуализации 3D-кролика"""

    def __init__(self):
        # Вершины стандартного кролика (стандартная модель Stanford Bunny)
        self.vertices = self.create_bunny_vertices()
        self.faces = self.create_bunny_faces()

        # Начальное положение и вращение
        self.translation = np.array([0.0, 0.0, 0.0])
        self.rotation = np.array([0.0, 0.0, 0.0])  # углы Эйлера (в радианах)
        self.scale = 1.0

        # Цвета
        self.face_colors = None

    def create_bunny_vertices(self):
        """Создаем упрощенную модель кролика"""
        # Упрощенная модель сферы с ушами
        vertices = []

        # Тело (сфера) - МЕНЬШЕ ТОЧЕК
        # Уменьшаем количество точек для сферы
        phi_steps = 6  # было 10
        theta_steps = 10  # было 20
        phi = np.linspace(0, np.pi, phi_steps)
        theta = np.linspace(0, 2 * np.pi, theta_steps)

        for p in phi:
            for t in theta:
                x = 0.5 * np.sin(p) * np.cos(t)
                y = 0.5 * np.sin(p) * np.sin(t) - 0.2  # Сдвигаем выше, чтобы уши были ближе
                z = 0.5 * np.cos(p)
                vertices.append([x, y, z])

        # Уши (конусы) - МЕНЬШЕ ТОЧЕК и БЛИЖЕ К ШАРУ
        # Уменьшаем количество точек для ушей
        height_steps = 4  # было 5
        angle_steps = 6  # было 10

        for i in range(2):
            sign = 1 if i == 0 else -1
            for h in np.linspace(0, 1, height_steps):
                for a in np.linspace(0, 2 * np.pi, angle_steps):
                    r = 0.08 * (1 - h)  # Чуть тоньше уши
                    x = sign * 0.15 + r * np.cos(a)  # Ближе к центру
                    y = 0.3 + h * 0.25  # НИЖЕ, чтобы были ближе к шару
                    z = r * np.sin(a)
                    vertices.append([x, y, z])

        return np.array(vertices)

    def create_bunny_faces(self):
        """Создаем грани для визуализации"""
        return None

    def get_transformed_vertices(self):
        """Получить трансформированные вершины"""
        # Масштабирование
        scaled_vertices = self.vertices * self.scale

        # Вращение (используем радианы для вычислений)
        rot_x = Matrix4f.rotation_x(self.rotation[0])
        rot_y = Matrix4f.rotation_y(self.rotation[1])
        rot_z = Matrix4f.rotation_z(self.rotation[2])

        # Трансляция
        trans = Matrix4f.translation(*self.translation)

        # Общая матрица преобразования
        transform = trans * rot_z * rot_y * rot_x

        # Преобразуем все вершины
        transformed = []
        for vertex in scaled_vertices:
            # Добавляем w=1 для однородных координат
            v = np.append(vertex, 1.0)
            v_transformed = transform * v
            transformed.append(v_transformed[:3])

        return np.array(transformed)

    def set_position(self, x, y, z):
        """Установить положение"""
        self.translation = np.array([x, y, z])

    def move_by(self, dx, dy, dz):
        """Переместить на указанные значения"""
        self.translation += np.array([dx, dy, dz])

    def set_rotation_rad(self, rx, ry, rz):
        """Установить вращение в радианах (для внутренних вычислений)"""
        self.rotation = np.array([rx, ry, rz])

    def set_rotation_deg(self, rx_deg, ry_deg, rz_deg):
        """Установить вращение в градусах"""
        self.rotation = np.array([radians(rx_deg), radians(ry_deg), radians(rz_deg)])

    def set_scale(self, scale):
        """Установить масштаб"""
        self.scale = scale

    def get_rotation_deg(self):
        """Получить текущие углы вращения в градусах"""
        return (
            math.degrees(self.rotation[0]),
            math.degrees(self.rotation[1]),
            math.degrees(self.rotation[2])
        )

    def get_position(self):
        """Получить текущее положение"""
        return self.translation.copy()


class TransformationVisualizer:
    """Класс для визуализации преобразований"""

    def __init__(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.bunny = Bunny3D()

        # Создаем сетку 1x2: левая часть - управление, правая часть - 3D визуализация
        gs = plt.GridSpec(1, 2, width_ratios=[1, 2])

        # Настройка области управления слева
        self.control_ax = plt.subplot(gs[0])
        self.control_ax.axis('off')  # Скрываем оси

        # Настройка 3D осей справа
        self.ax = self.fig.add_subplot(gs[1], projection='3d')

        # Сохраняем начальный вид камеры
        self.initial_view = {
            'elev': 30,
            'azim': -60,
            'dist': 10
        }

        # Устанавливаем начальный вид
        self.ax.view_init(elev=self.initial_view['elev'], azim=self.initial_view['azim'])

        # Настройка всех элементов управления на левой панели
        self.setup_control_panel()

        # Начальная визуализация
        self.update_plot()

        # Настройка 3D графика
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Bunny Transformation Visualizer')
        self.ax.set_xlim([-5, 5])
        self.ax.set_ylim([-5, 5])
        self.ax.set_zlim([-5, 5])
        self.ax.grid(True)

        # Добавляем оси координат
        self.draw_coordinate_axes()

    def draw_coordinate_axes(self):
        """Рисуем оси координат"""
        length = 1.5
        # Ось X (красная)
        self.ax.quiver(0, 0, 0, length, 0, 0, color='r', arrow_length_ratio=0.1, linewidth=2)
        # Ось Y (зеленая)
        self.ax.quiver(0, 0, 0, 0, length, 0, color='g', arrow_length_ratio=0.1, linewidth=2)
        # Ось Z (синяя)
        self.ax.quiver(0, 0, 0, 0, 0, length, color='b', arrow_length_ratio=0.1, linewidth=2)

        # Подписи осей
        self.ax.text(length, 0, 0, 'X', color='r', fontsize=12)
        self.ax.text(0, length, 0, 'Y', color='g', fontsize=12)
        self.ax.text(0, 0, length, 'Z', color='b', fontsize=12)

    def setup_control_panel(self):
        """Настройка всей панели управления слева"""
        # Размеры и отступы (уменьшены размеры кнопок)
        left_margin = 0.05
        top_margin = 0.95
        text_width = 0.25
        text_height = 0.04  # Уменьшена высота текстовых полей
        button_width = 0.25
        button_height = 0.045  # Уменьшена высота кнопок
        vertical_spacing = 0.05  # Уменьшено расстояние между элементами

        current_y = top_margin

        # Заголовок панели управления
        title_ax = plt.axes([left_margin, current_y - 0.02, 0.9, 0.04])
        title_ax.axis('off')
        title_ax.text(0.5, 0.5, 'Control Panel',
                      horizontalalignment='center',
                      verticalalignment='center',
                      fontsize=12, fontweight='bold')

        current_y -= 0.06

        # Раздел: Положение объекта
        section_ax = plt.axes([left_margin, current_y - 0.02, 0.9, 0.03])
        section_ax.axis('off')
        section_ax.text(0, 0.5, 'Position:',
                        horizontalalignment='left',
                        verticalalignment='center',
                        fontsize=10, fontweight='bold')

        current_y -= 0.04

        # Позиция X
        label_x_ax = plt.axes([left_margin, current_y, 0.12, text_height])
        label_x_ax.axis('off')
        label_x_ax.text(0.5, 0.5, 'X:',
                        horizontalalignment='right',
                        verticalalignment='center',
                        fontsize=9)

        text_x_ax = plt.axes([left_margin + 0.13, current_y, text_width, text_height])
        self.text_px = TextBox(text_x_ax, '', initial="0.0")
        self.text_px.on_submit(lambda text: self.update_object_position('X', text))

        current_y -= vertical_spacing

        # Позиция Y
        label_y_ax = plt.axes([left_margin, current_y, 0.12, text_height])
        label_y_ax.axis('off')
        label_y_ax.text(0.5, 0.5, 'Y:',
                        horizontalalignment='right',
                        verticalalignment='center',
                        fontsize=9)

        text_y_ax = plt.axes([left_margin + 0.13, current_y, text_width, text_height])
        self.text_py = TextBox(text_y_ax, '', initial="0.0")
        self.text_py.on_submit(lambda text: self.update_object_position('Y', text))

        current_y -= vertical_spacing

        # Позиция Z
        label_z_ax = plt.axes([left_margin, current_y, 0.12, text_height])
        label_z_ax.axis('off')
        label_z_ax.text(0.5, 0.5, 'Z:',
                        horizontalalignment='right',
                        verticalalignment='center',
                        fontsize=9)

        text_z_ax = plt.axes([left_margin + 0.13, current_y, text_width, text_height])
        self.text_pz = TextBox(text_z_ax, '', initial="0.0")
        self.text_pz.on_submit(lambda text: self.update_object_position('Z', text))

        current_y -= 0.06

        # Раздел: Вращение объекта
        section_rot_ax = plt.axes([left_margin, current_y - 0.02, 0.9, 0.03])
        section_rot_ax.axis('off')
        section_rot_ax.text(0, 0.5, 'Rotation (degrees):',
                            horizontalalignment='left',
                            verticalalignment='center',
                            fontsize=10, fontweight='bold')

        current_y -= 0.04

        # Вращение X
        label_rx_ax = plt.axes([left_margin, current_y, 0.12, text_height])
        label_rx_ax.axis('off')
        label_rx_ax.text(0.5, 0.5, 'X:',
                         horizontalalignment='right',
                         verticalalignment='center',
                         fontsize=9)

        text_rx_ax = plt.axes([left_margin + 0.13, current_y, text_width, text_height])
        self.text_rx = TextBox(text_rx_ax, '', initial="0")
        self.text_rx.on_submit(lambda text: self.update_object_rotation('X', text))

        current_y -= vertical_spacing

        # Вращение Y
        label_ry_ax = plt.axes([left_margin, current_y, 0.12, text_height])
        label_ry_ax.axis('off')
        label_ry_ax.text(0.5, 0.5, 'Y:',
                         horizontalalignment='right',
                         verticalalignment='center',
                         fontsize=9)

        text_ry_ax = plt.axes([left_margin + 0.13, current_y, text_width, text_height])
        self.text_ry = TextBox(text_ry_ax, '', initial="0")
        self.text_ry.on_submit(lambda text: self.update_object_rotation('Y', text))

        current_y -= vertical_spacing

        # Вращение Z
        label_rz_ax = plt.axes([left_margin, current_y, 0.12, text_height])
        label_rz_ax.axis('off')
        label_rz_ax.text(0.5, 0.5, 'Z:',
                         horizontalalignment='right',
                         verticalalignment='center',
                         fontsize=9)

        text_rz_ax = plt.axes([left_margin + 0.13, current_y, text_width, text_height])
        self.text_rz = TextBox(text_rz_ax, '', initial="0")
        self.text_rz.on_submit(lambda text: self.update_object_rotation('Z', text))

        current_y -= 0.06

        # Раздел: Масштаб
        section_scale_ax = plt.axes([left_margin, current_y - 0.02, 0.9, 0.03])
        section_scale_ax.axis('off')
        section_scale_ax.text(0, 0.5, 'Scale:',
                              horizontalalignment='left',
                              verticalalignment='center',
                              fontsize=10, fontweight='bold')

        current_y -= 0.04

        label_scale_ax = plt.axes([left_margin, current_y, 0.12, text_height])
        label_scale_ax.axis('off')
        label_scale_ax.text(0.5, 0.5, 'Value:',
                            horizontalalignment='right',
                            verticalalignment='center',
                            fontsize=9)

        text_scale_ax = plt.axes([left_margin + 0.13, current_y, text_width, text_height])
        self.text_scale = TextBox(text_scale_ax, '', initial="1.0")
        self.text_scale.on_submit(self.update_object_scale)

        current_y -= 0.08

        # Раздел: Управление камерой
        section_camera_ax = plt.axes([left_margin, current_y - 0.02, 0.9, 0.03])
        section_camera_ax.axis('off')
        section_camera_ax.text(0, 0.5, 'Camera Control:',
                               horizontalalignment='left',
                               verticalalignment='center',
                               fontsize=10, fontweight='bold')

        current_y -= 0.04

        # Кнопка сброса вида камеры
        reset_view_ax = plt.axes([left_margin, current_y, button_width, button_height])
        self.reset_view_button = Button(reset_view_ax, 'Reset View')
        self.reset_view_button.on_clicked(self.reset_camera_view)

        current_y -= vertical_spacing + 0.01

        # Кнопки стандартных видов камеры (в два столбца)
        view_x_ax = plt.axes([left_margin, current_y, button_width / 2 - 0.01, button_height])
        self.view_x_button = Button(view_x_ax, 'View X')
        self.view_x_button.on_clicked(lambda event: self.set_camera_view(0, 90))

        view_y_ax = plt.axes([left_margin + button_width / 2 + 0.01, current_y, button_width / 2 - 0.01, button_height])
        self.view_y_button = Button(view_y_ax, 'View Y')
        self.view_y_button.on_clicked(lambda event: self.set_camera_view(0, 0))

        current_y -= vertical_spacing

        view_z_ax = plt.axes([left_margin, current_y, button_width / 2 - 0.01, button_height])
        self.view_z_button = Button(view_z_ax, 'View Z')
        self.view_z_button.on_clicked(lambda event: self.set_camera_view(90, 0))

        top_view_ax = plt.axes(
            [left_margin + button_width / 2 + 0.01, current_y, button_width / 2 - 0.01, button_height])
        self.top_view_button = Button(top_view_ax, 'Top')
        self.top_view_button.on_clicked(lambda event: self.set_camera_view(90, -90))

        current_y -= vertical_spacing

        front_view_ax = plt.axes([left_margin, current_y, button_width / 2 - 0.01, button_height])
        self.front_view_button = Button(front_view_ax, 'Front')
        self.front_view_button.on_clicked(lambda event: self.set_camera_view(0, -90))

        side_view_ax = plt.axes(
            [left_margin + button_width / 2 + 0.01, current_y, button_width / 2 - 0.01, button_height])
        self.side_view_button = Button(side_view_ax, 'Side')
        self.side_view_button.on_clicked(lambda event: self.set_camera_view(0, 0))

        current_y -= 0.08

        # Раздел: Управление объектом
        section_object_ax = plt.axes([left_margin, current_y - 0.02, 0.9, 0.03])
        section_object_ax.axis('off')
        section_object_ax.text(0, 0.5, 'Object Control:',
                               horizontalalignment='left',
                               verticalalignment='center',
                               fontsize=10, fontweight='bold')

        current_y -= 0.04

        # Кнопка сброса всех преобразований
        reset_ax = plt.axes([left_margin, current_y, button_width, button_height])
        self.reset_button = Button(reset_ax, 'Reset All')
        self.reset_button.on_clicked(self.reset_object_transformations)

        current_y -= vertical_spacing

        # Кнопка сброса вращения
        reset_rot_ax = plt.axes([left_margin, current_y, button_width, button_height])
        self.reset_rot_button = Button(reset_rot_ax, 'Reset Rotation')
        self.reset_rot_button.on_clicked(self.reset_object_rotation)

        current_y -= vertical_spacing

        # Кнопка анимации
        anim_ax = plt.axes([left_margin, current_y, button_width, button_height])
        self.anim_button = Button(anim_ax, 'Animate')
        self.anim_button.on_clicked(self.start_animation)

    def update_object_position(self, axis, text):
        """Обновить позицию объекта из текстового поля"""
        try:
            # Парсим значение из текстового поля
            value = float(text.strip()) if text.strip() else 0.0

            # Получаем текущее положение объекта
            x, y, z = self.bunny.get_position()

            # Обновляем нужную координату
            if axis == 'X':
                x = value
            elif axis == 'Y':
                y = value
            elif axis == 'Z':
                z = value

            # Устанавливаем новое положение объекта
            self.bunny.set_position(x, y, z)
            self.update_plot()
        except ValueError:
            # В случае ошибки возвращаем текущие значения
            self.update_position_display()

    def update_object_rotation(self, axis, text):
        """Обновить вращение объекта из текстового поля"""
        try:
            # Парсим значение из текстового поля
            angle = float(text.strip()) if text.strip() else 0

            # Получаем текущие углы объекта
            rx, ry, rz = self.bunny.get_rotation_deg()

            # Обновляем нужный угол
            if axis == 'X':
                rx = angle
            elif axis == 'Y':
                ry = angle
            elif axis == 'Z':
                rz = angle

            # Устанавливаем вращение объекта
            self.bunny.set_rotation_deg(rx, ry, rz)
            self.update_plot()
        except ValueError:
            # В случае ошибки возвращаем текущие значения
            self.update_rotation_display()

    def update_object_scale(self, text):
        """Обновить масштаб объекта из текстового поля"""
        try:
            # Парсим значение из текстового поля
            scale = float(text.strip()) if text.strip() else 1.0

            # Устанавливаем масштаб объекта
            self.bunny.set_scale(scale)
            self.update_plot()
        except ValueError:
            # В случае ошибки возвращаем текущее значение
            self.text_scale.set_val(f"{self.bunny.scale:.2f}")

    def update_position_display(self):
        """Обновить отображение позиции объекта в текстовых полях"""
        x, y, z = self.bunny.get_position()

        # Обновляем текстовые поля (временно отключаем события)
        self.text_px.eventson = False
        self.text_py.eventson = False
        self.text_pz.eventson = False

        self.text_px.set_val(f"{x:.2f}")
        self.text_py.set_val(f"{y:.2f}")
        self.text_pz.set_val(f"{z:.2f}")

        self.text_px.eventson = True
        self.text_py.eventson = True
        self.text_pz.eventson = True

    def update_rotation_display(self):
        """Обновить отображение углов вращения объекта в текстовых полях"""
        rx, ry, rz = self.bunny.get_rotation_deg()

        # Обновляем текстовые поля (временно отключаем события)
        self.text_rx.eventson = False
        self.text_ry.eventson = False
        self.text_rz.eventson = False

        self.text_rx.set_val(f"{rx:.1f}")
        self.text_ry.set_val(f"{ry:.1f}")
        self.text_rz.set_val(f"{rz:.1f}")

        self.text_rx.eventson = True
        self.text_ry.eventson = True
        self.text_rz.eventson = True

    def reset_camera_view(self, event):
        """Сбросить вид камеры к начальному"""
        self.ax.view_init(elev=self.initial_view['elev'], azim=self.initial_view['azim'])
        self.update_plot()

    def set_camera_view(self, elev, azim):
        """Установить определенный вид камеры"""
        self.ax.view_init(elev=elev, azim=azim)
        self.update_plot()

    def reset_object_transformations(self, event):
        """Сбросить все преобразования объекта"""
        # Сброс положения
        self.bunny.set_position(0, 0, 0)

        # Сброс вращения объекта
        self.bunny.set_rotation_deg(0, 0, 0)

        # Сброс масштаба
        self.bunny.set_scale(1.0)

        # Обновление отображения
        self.update_position_display()
        self.update_rotation_display()
        self.text_scale.set_val("1.0")
        self.update_plot()

    def reset_object_rotation(self, event):
        """Сбросить только вращение объекта"""
        self.bunny.set_rotation_deg(0, 0, 0)
        self.update_rotation_display()
        self.update_plot()

    def start_animation(self, event):
        """Запустить анимацию вращения объекта"""
        # Сохраняем начальные значения
        initial_rot = self.bunny.rotation.copy()

        # Функция для анимации
        def animate(frame):
            # Добавляем вращение в радианах
            self.bunny.set_rotation_rad(
                initial_rot[0] + frame * 0.1,
                initial_rot[1] + frame * 0.05,
                initial_rot[2] + frame * 0.02
            )

            # Обновляем отображение углов
            self.update_rotation_display()

            self.update_plot()
            return self.ax

        # Создаем анимацию
        ani = animation.FuncAnimation(
            self.fig, animate, frames=100,
            interval=50, blit=False, repeat=True
        )
        plt.draw()

    def update_plot(self):
        """Обновление графика"""
        self.ax.clear()

        # Рисуем оси координат
        self.draw_coordinate_axes()

        # Получаем трансформированные вершины
        vertices = self.bunny.get_transformed_vertices()

        # Рисуем кролика как облако точек с фиолетово-бардовым градиентом
        colors = vertices[:, 1]  # Используем Y-координату для цвета

        # Изменено на plasma - фиолетово-бардовый градиент
        scatter = self.ax.scatter(
            vertices[:, 0], vertices[:, 1], vertices[:, 2],
            c=colors, cmap='plasma', s=20, alpha=0.8  # Изменено с 'viridis' на 'plasma'
        )

        # Настройка графика
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Bunny Transformation Visualizer')

        # Автоматически устанавливаем пределы осей в зависимости от положения объекта
        x, y, z = self.bunny.get_position()
        margin = 5
        self.ax.set_xlim([x - margin, x + margin])
        self.ax.set_ylim([y - margin, y + margin])
        self.ax.set_zlim([z - margin, z + margin])

        self.ax.grid(True)

        # Обновляем холст
        self.fig.canvas.draw_idle()


def test_matrix_operations():
    """Тестирование матричных операций"""
    print("Тестирование матричных операций:")
    print("=" * 50)

    # Тест трансляции
    trans = Matrix4f.translation(1, 2, 3)
    print(f"Матрица трансляции:\n{trans}")

    # Тест вращения вокруг X (45 градусов)
    rot_x = Matrix4f.rotation_x(np.pi / 4)
    print(f"Матрица вращения вокруг X (45°):\n{rot_x}")

    # Тест вращения вокруг произвольной оси (60 градусов)
    rot_axis = Matrix4f.rotation_axis(np.pi / 3, 1, 1, 0)
    print(f"Матрица вращения вокруг оси (1,1,0) (60°):\n{rot_axis}")

    # Тест умножения матриц
    combined = trans * rot_x
    print(f"Комбинированная матрица (трансляция * вращение X):\n{combined}")

    # Тест преобразования точки
    point = np.array([1.0, 0.0, 0.0, 1.0])
    transformed_point = combined * point
    print(f"Точка (1,0,0,1) после преобразования:\n{transformed_point}")


def main():
    """Основная функция"""
    print("3D Transformation Visualizer")
    print("=" * 50)
    print("Управление объектом (левая панель):")
    print("- Введите координаты X, Y, Z в соответствующие поля")
    print("- Введите углы вращения в градусах")
    print("- Введите значение масштаба (по умолчанию 1.0)")
    print("- Используйте кнопки для управления камерой")
    print("- Используйте кнопки для сброса преобразований")
    print("- Нажмите 'Animate' для запуска анимации")
    print("\n3D Визуализация (правая панель):")
    print("- Вращайте сцену мышкой для изменения угла обзора")
    print("- Объект отображается в правой части программы")
    print("\nВсе элементы управления расположены на левой панели для удобства")
    print("\nИзменения в модели кролика:")
    print("- Уши расположены ближе к шару (телу)")
    print("- Уменьшено количество точек в модели")
    print("- Более компактная и быстрая визуализация")
    print("\nНовый градиент цвета:")
    print("- Точки зайчика окрашены в фиолетово-бардовый градиент")

    # Запускаем тесты
    test_matrix_operations()

    # Запускаем визуализатор
    visualizer = TransformationVisualizer()
    plt.show()


if __name__ == "__main__":
    main()