import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils

import flet as ft

import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from tkinter import simpledialog
import re

def main(page: ft.Page):
    page.title = "Trabalho PDI"
    page.vertical_alignment = ft.MainAxisAlignment.START
    page.horizontal_alignment = ft.CrossAxisAlignment.START
    page.window_width = 550
    page.window_height = 975

    ##########################################################################

    # FILTERING FUNCTIONS

    def plot_histogram(image):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        y = img_yuv[:,:,0]
        y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)

        # Calcula o histograma da imagem
        hist = cv2.calcHist([y], [0], None, [256], [0,256])

        # Plota o histograma em barras
        plt.bar(range(256), hist.ravel(), width=1, color='cyan')
        plt.title('Histograma da Imagem(Grayscale)')
        plt.xlabel('Intensidade')
        plt.ylabel('Frequência')
        plt.savefig('histo.jpg')

    def apply_cb_balance_plus(img, cb):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        v_arr = (re.findall(r'\d+', cb))
        v = int(v_arr[0])

        img[:, :, 1] = cv2.add(img[:, :, 1], v)

        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        
        return img
    
    def apply_cr_balance_plus(img, cr):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        v_arr = (re.findall(r'\d+', cr))
        v = int(v_arr[0])

        img[:, :, 2] = cv2.add(img[:, :, 2], v)

        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

        return img

    def apply_cb_balance_less(img, cb):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        v_arr = (re.findall(r'\d+', cb))
        v = int(v_arr[0])

        img[:, :, 1] = cv2.subtract(img[:, :, 1], v)

        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        
        return img
    
    def apply_cr_balance_less(img, cr):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        v_arr = (re.findall(r'\d+', cr))
        v = int(v_arr[0])

        img[:, :, 2] = cv2.subtract(img[:, :, 2], v)

        img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

        return img

    def adjust_gamma(image, gamma):
        v_arr = (re.findall(r'\d+', gamma))
        v = float(v_arr[0])

        inv_gamma = 1.0 / v
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")

        return cv2.LUT(image, table)

    def equalize_histogram(img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        y = img_yuv[:,:,0]

        image = cv2.equalizeHist(y)

        img_yuv[:,:,0] = image

        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    def make_lut_u():
        return np.array([[[i,255-i,0] for i in range(256)]],dtype=np.uint8)

    def make_lut_v():
        return np.array([[[0,255-i,i] for i in range(256)]],dtype=np.uint8)

    def open_y(img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        y = img_yuv[:,:,0]
        y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)

        cv2.imshow('Canal luminosidade', y)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def open_cb(img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        
        _, u, _ = cv2.split(img_yuv)
        lut_u = make_lut_u()
        u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
        u_mapped = cv2.LUT(u, lut_u)

        cv2.imshow('Canal azul', u_mapped)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def open_cr(img):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        _, _, v = cv2.split(img_yuv)
        lut_v = make_lut_v()
        v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
        v_mapped = cv2.LUT(v, lut_v)

        cv2.imshow('Canal vermelho', v_mapped)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def open_original():
        cv2.imshow('Imagem Original', cv2.imread('original.jpg'))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def open_filtered():
        cv2.imshow('Imagem Filtrada', cv2.imread('filtered.jpg'))

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ##########################################################################

    def get_current_img():
        return cv2.imread('original.jpg')

    def get_new_img():
        f_types = [('Jpg Files', '*.jpg')]
        filename = filedialog.askopenfilename(filetypes=f_types)
        cv2.imwrite('./original.jpg', cv2.imread(filename))

        page.update()

    def set_img():
        cv2.imwrite('./original.jpg', cv2.imread('./filtered.jpg'))
        
        page.update()

    def write_filtered(img):
        cv2.imwrite('./filtered.jpg', img)
        cv2.imshow('Imagem Filtrada', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def slider_changed(e, t):
        t.value = f"Intensidade do filtro: {int(e.control.value)}"

        page.update()

    def slider_changed_gamma(e, t):
        t.value = f"Intensidade de gamma: {round(e.control.value, 2)}"

        page.update()

    t1 = ft.Text()
    t2 = ft.Text()
    t3 = ft.Text()

    page.add(
        ft.Column(
        [
            ft.Row(
                [
                    ft.Container(
                        content=ft.Text("Ajustes em Y'CbCr", size=40),
                        margin=1,
                        padding=10,
                        alignment=ft.alignment.center,
                        width=360,
                        height=100,
                        border_radius=10,
                        ink=True,
                    ),
                    ft.Container(
                        content=ft.Icon(name=ft.icons.FILE_UPLOAD),
                        margin=1,
                        padding=10,
                        alignment=ft.alignment.center,
                        bgcolor=ft.colors.BLUE_400,
                        width=70,
                        height=60,
                        border_radius=10,
                        ink=True,
                        on_click=lambda e: get_new_img(),
                    ),
                    ft.Container(
                        content=ft.Icon(name=ft.icons.FILE_DOWNLOAD),
                        margin=1,
                        padding=10,
                        alignment=ft.alignment.center,
                        bgcolor=ft.colors.BLUE_400,
                        width=70,
                        height=60,
                        border_radius=10,
                        ink=True,
                        on_click=lambda e: set_img(),
                    ),
                ],
            ),
            ft.Row(
                [
                ft.Container(
                    content=ft.Text("Abrir original"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.BLUE_400,
                    width=240,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: open_original(),
                    ),
                ft.Container(
                    content=ft.Text("Abrir filtrada"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.BLUE_400,
                    width=240,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: open_filtered(),
                    ),
                ]
            ),
            ft.Row(
                [
                ft.Container(
                    content=ft.Text("Ajuste gamma"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.BLUE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(adjust_gamma(get_current_img(), t3.value)),
                    ),
                ft.Container(
                    content=ft.Text("Ajuste luz"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.BLUE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(equalize_histogram(get_current_img())),
                    ),
                ft.Container(
                    content=ft.Icon(name=ft.icons.ASSESSMENT_OUTLINED),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.BLUE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: plot_histogram(get_current_img()),
                    ),
                ]
            ),
            ft.Row(
                [
                ft.Container(
                    content=ft.Text("Canal Y"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.BLUE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: open_y(get_current_img()),
                    ),
                ft.Container(
                    content=ft.Text("Canal Cb"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.BLUE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: open_cb(get_current_img()),
                    ),
                ft.Container(
                    content=ft.Text("Canal Cr"),
                    margin=10,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.BLUE_400,
                    width=150,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: open_cr(get_current_img()),
                    ),
                ]
            ),
            ft.Row(
                [
                ft.Container(
                    content=ft.Text("Ajustes em Cb", size=20),
                    margin=1,
                    padding=10,
                    alignment=ft.alignment.center_right,
                    width=360,
                    height=100,
                    border_radius=10,
                    ink=True,
                    ),
                ft.Container(
                    content=ft.Icon(name=ft.icons.ADD),
                    margin=1,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.BLUE_400,
                    width=70,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_cb_balance_plus(get_current_img(), t1.value)),
                    ),
                ft.Container(
                    content=ft.Icon(name=ft.icons.REMOVE),
                    margin=1,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.BLUE_400,
                    width=70,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_cb_balance_less(get_current_img(), t1.value)),
                    ),
                ]
            ),
            ft.Row(
                [
                ft.Container(
                    content=ft.Text("Ajustes em Cr", size=20),
                    margin=1,
                    padding=10,
                    alignment=ft.alignment.center_right,
                    width=360,
                    height=100,
                    border_radius=10,
                    ink=True,
                    ),
                ft.Container(
                    content=ft.Icon(name=ft.icons.ADD),
                    margin=1,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.BLUE_400,
                    width=70,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_cr_balance_plus(get_current_img(), t2.value)),
                    ),
                ft.Container(
                    content=ft.Icon(name=ft.icons.REMOVE),
                    margin=1,
                    padding=10,
                    alignment=ft.alignment.center,
                    bgcolor=ft.colors.BLUE_400,
                    width=70,
                    height=60,
                    border_radius=10,
                    ink=True,
                    on_click=lambda e: write_filtered(apply_cr_balance_less(get_current_img(), t2.value)),
                    ),
                ]
            ),
        ],
        ),
        ft.Text("Valor para o canal Cb(azul): "),
        ft.Slider(min=-127, max=128, divisions=100, label="{value}", on_change=lambda e: slider_changed(e, t1)), t1),

    page.add(    
        ft.Text("Valor para o canal Cr(vermelho): "),
        ft.Slider(min=-127, max=128, divisions=100, label="{value}", on_change=lambda e: slider_changed(e, t2)), t2),

    page.add(    
        ft.Text("Ajuste de gamma: "),
        ft.Slider(min=1, max=5, divisions=100, label="{value}", on_change=lambda e: slider_changed_gamma(e, t3)), t3)

ft.app(target=main)