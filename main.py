import pygame
from pygame import *
import torch
import numpy as np

from model import LeNet33
from preprocessor import preproc

pygame.init()

# создание модели с весами
model_1 = torch.load('lenet33_model.pth', map_location=torch.device('cpu'))
model_1.eval()


# функция предсказания значения
def pred(tensor):
    predict = torch.nn.functional.softmax(model_1.forward(tensor), dim=1) * 100
    predict = torch.round(predict * 10) / 10
    return predict


# размеры основных элементов
window_width = 660
window_height = 560
sidebar_width = 100

# основные цвета
WHITE = (254, 254, 254)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)

# размеры кисти(радиус круга, которым рисуем)
brush_size = 30

# вид основного окна программы
sc = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Digit-Recognizer')
# pygame.display.set_icon()

# в окне рисования пользователь рисует цифру
drawing_window = pygame.Surface((window_width - 100, window_height))
drawing_window.fill(WHITE)

# в сайдбаре отображаются вероятности получаемые от модели
sidebar = pygame.Surface((sidebar_width, window_height))
sidebar.fill(GRAY)

# окошки для отображения вероятностей
value_boxes = []
for i in range(10):
    value_box = pygame.Rect(10, i * 55 + 10, sidebar_width - 20, 40)
    value_boxes.append(value_box)

pygame.display.update()

# настройка FPS
FPS = 60
clock = pygame.time.Clock()

# отслеживание состояния кнопок мыши
left_button_pressed = False
right_button_pressed = False

# основной цикл программы
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # выход из приложения
        elif event.type == MOUSEBUTTONDOWN:
            if event.button == 1:  # левая кнопка мыши
                left_button_pressed = True
            elif event.button == 3:  # правая кнопка мыши
                right_button_pressed = True
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:  # левая кнопка мыши
                left_button_pressed = False
            elif event.button == 3:  # правая кнопка мыши
                right_button_pressed = False
        elif event.type == MOUSEMOTION:
            if left_button_pressed or right_button_pressed:
                x, y = event.pos  # получаем координаты мыши
                if x < 0:  # установка ограничений, иначе при нажатой клавише мыши
                    x = 0  # координаты могут выйти за пределы окна
                if x > 560:
                    x = 560
                if y < 0:
                    y = 0
                if y > 560:
                    y = 560
                if 0 <= x <= 560 and 0 <= y <= 560:
                    color = BLACK if left_button_pressed else WHITE  # Определяем цвет
                    brush_thickness = brush_size if left_button_pressed else brush_size * 1.5
                    pygame.draw.circle(drawing_window, color, (x, y), brush_thickness)

        # получение изображения в виде массива
        array = pygame.surfarray.array3d(drawing_window)

        if np.min(array) > 0:  # проверка, нарисовано ли хоть что-то в окне
            for i, box in enumerate(value_boxes):
                prob = 0
                value = str(i) + ': ' + str(prob) + '%'
                pygame.draw.rect(sidebar, WHITE, box)
                font = pygame.font.Font(None, 24)
                text = font.render(f"{value}", True, BLACK)
                text_rect = text.get_rect(center=box.center)
                sidebar.blit(text, text_rect)
        else:
            array = preproc(array)  # обработка массива
            result = pred(array)  # получение предикта
            for i, box in enumerate(value_boxes):
                prob = round(result[0][i].item(), 1)
                x = prob / 100
                R = round(255 * (1 - x))
                G = round(255 * x)
                value = str(i) + ': ' + str(prob) + '%'
                pygame.draw.rect(sidebar, (R, G, 0), box)
                font = pygame.font.Font(None, 24)
                text = font.render(f"{value}", True, BLACK)
                text_rect = text.get_rect(center=box.center)
                sidebar.blit(text, text_rect)

        # Отображение поверхностей на основном экране
        sc.blit(drawing_window, (0, 0))
        sc.blit(sidebar, (window_width - sidebar_width, 0))

        pygame.display.update()

    clock.tick(FPS)
