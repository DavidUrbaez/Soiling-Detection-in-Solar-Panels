from ipywidgets import interact, interactive
import ipywidgets as widgets
from ipywidgets import Layout
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage import io, color
import os
import sys
import glob


from utils.funciones_utiles import *



##############################################################
# Create GUI

def create_GUI(images=None):
    
    for file in os.listdir('./res/img_with_transform/'):
        if file.endswith('.jpg'):
            os.remove('./res/img_with_transform/'+file)

    # Se crean las pestañas

    out1 = widgets.Output()
    out2 = widgets.Output()
    out3 = widgets.Output()
    out4 = widgets.Output()
    out5 = widgets.Output()
    out6 = widgets.Output()
    out7 = widgets.Output()

    box_layout = Layout(display='flex',
                        flex_flow='column')

    # Se crean los "interact" y se definen (max_v=0, min_v=0, step=0) Para cada transformación

    with out1:
        show_interact('suma', 255, 0, 1, lineal=True,images=images)        

    with out2:
        show_interact('resta', 255, 0, 1, lineal=True,images=images)

    with out3:
        show_interact('multiplicacion', 5, 1, 1, lineal=True,images=images)

    with out4:
        show_interact('division', 1, 0, 0.01, lineal=True,images=images)

    with out5:
        show_interact('T. gamma', 2, 0, 0.1, lineal=False,images=images)

    with out6:
        show_interact('Ec. histograma', lineal=False,images=images)

    with out7:
        show_interact('Exp. histograma', lineal=False,images=images)


    ####################################
    # Se crean los tabs de los widgets y se les agrega el titulo

    tab = widgets.Tab(children=[out1, out2, out3, out4, out5, out6, out7])
    tab.set_title(0, 'Suma')
    tab.set_title(1, 'Resta')
    tab.set_title(2, 'Multiplicación')
    tab.set_title(3, 'División')
    tab.set_title(4, 'T. gamma')
    tab.set_title(5, 'Ec. histograma')
    tab.set_title(6, 'Exp. histograma')

    return tab


##############################################################
# Show Interact

def show_interact(function, max_v=0, min_v=0, step=0, lineal=True,images=None):


    box_layout = Layout(display='flex',
                        flex_flow='column')

    _ = interact(
        analisis_espacio_color,
        color_space=widgets.ToggleButtons(
                                        options=['RGB','CMY','YIQ','YUB','HSL','HSV','LAB','XYZ','HLS'],
                                        description='Escoge el espacio de color a usar:',
                                        disabled=False,
                                        button_style='info',
                                        tooltips=['modelo aditivo', 'modelo sustractivo',
                                                  'separa la información de intensidad o luminancia',
                                                  'ancho de banda reducido para los componentes de crominancia',
                                                  'fácilmente interpretables y legibles por un humano ...',
                                                  '... métodos en los que la representación del componente de color no es lineal',
                                                  'L -> información sobre luminosidad, a* y b* -> información de color',
                                                 'sensores de color (XYZ)'],
                                        layout=box_layout,
                                        style= {'description_width': '200px'}
                                        ),
        channel=widgets.RadioButtons(description="Escoge el canal a visualizar:", options=[1, 2, 3],style= {'description_width': 'auto'}, ),

        a = widgets.FloatSlider(value=1,
                                    min=min_v,
                                    max=max_v,
                                    step=step,
                                    description='canal 1:' if function!='T. gamma' else 'a: ',
                                    disabled=False,
                                    continuous_update=False,
                                    orientation='horizontal',
                                    readout=True,
                                    readout_format='.1f',
                                    layout=Layout(display='none') if (not lineal and function!='T. gamma') else Layout()
                                ),
        b = widgets.FloatSlider(value=1,
                                    min=min_v,
                                    max=max_v,
                                    step=step,
                                    description='canal 2:' if function!='T. gamma' else 'gamma: ',
                                    disabled=False,
                                    continuous_update=False,
                                    orientation='horizontal',
                                    readout=True,
                                    readout_format='.1f',
                                    layout=Layout(display='none') if (not lineal and function!='T. gamma') else Layout()
                                ),
        c = widgets.FloatSlider(value=1,
                                    min=min_v,
                                    max=max_v,
                                    step=step,
                                    description='canal 3:',
                                    disabled=False,
                                    continuous_update=False,
                                    orientation='horizontal',
                                    readout=True,
                                    readout_format='.1f',
                                    layout=Layout(display='none') if not lineal else Layout()
                                ),
        f = widgets.Text(
                                value=function,
                                placeholder='Type something',
                                description='String:',
                                disabled=False,
                                layout=Layout(display='none')
                            ),
        lineal = widgets.Text(options=['RGB','CMY','YIQ','YUB','HSL','HSV','LAB','XYZ','HLS'],
                                value=str(lineal),
                                placeholder='Type something',
                                description='String:',
                                disabled=False,
                                layout=Layout(display='none')
                            ),
        
        img=widgets.Dropdown(
            options=[str(image) for image in images], value=str(images[0]), description="Escoger imagen:",
            style= {'description_width': 'auto'}
        ),
        
        save = widgets.ToggleButtons(value=None,
                                        options=['Guardar 3 canales','Guardar 1 canal'],
                                        description='Cómo quieres guardar la imagen:',
                                        disabled=False,
                                        button_style='warning',
                                        tooltips=[''],
                                        layout=box_layout,
                                        style= {'description_width': '200px'}
                                        ),
        hold_on = widgets.ToggleButtons(value=None,
                                        options=['Hold on', 'Refresh'],
                                        icon='refresh',
                                        description='.',
                                        disabled=False,
                                        button_style='success',
                                        tooltips=[''],
                                        layout=box_layout,
                                        style= {'position': 'right'}
                                        ),
    );

##############################################################
# Show Interact

def analisis_espacio_color(color_space, channel, img, a, b, c, f, lineal, save, hold_on):
    
    img = cv2.imread(img)
    img_space = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    img_tittle = color_space
    channel_tittle = color_space[channel - 1]

    color_space, img_space = get_color_space(img, color_space, img_space)

    img_transform = cv2.imread('./img_with_transform/img_transform.jpg')
    
    if img_transform is not None:
        img_space = img_transform
    
    if lineal == 'True':
        img_space = apply_linear_function(img_space, f, [a, b, c])

    elif lineal == 'False':
        img_space = apply_non_linear_function(img_space, f, [a, b])

    img_channel = img_space[:, :, channel - 1]

    fig, arreglo_plots = plt.subplots(1, 3, figsize=(20, 5))
    colors = ('r', 'g', 'b')

    arreglo_plots[0].set_title(img_tittle)
    arreglo_plots[0].imshow(img_space)

    arreglo_plots[1].set_title('Canal ' + channel_tittle)
    arreglo_plots[1].imshow(img_channel, cmap="gray")

    arreglo_plots[2].set_title('Histograma Canal ' + channel_tittle)
    img_array = img_channel.ravel()

    arreglo_plots[2].hist(img_array, histtype='step', bins=255,
                          range=(0.0, 255.0), density=True, color=colors[channel - 1])
    plt.show()

    if save == 'Guardar 3 canales':
        cv2.imwrite(os.path.join("results/", img_tittle + '.jpg'), img_space)
    elif save == 'Guardar 1 canal':
        cv2.imwrite(os.path.join("results/", channel_tittle + '.jpg'), img_channel)
        
    if hold_on == 'Hold on':
        cv2.imwrite('./img_with_transform/img_transform.jpg',img_space)
