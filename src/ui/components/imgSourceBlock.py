import flet as ft
from typing import Literal


@ft.control
class ImgSourceBlock(ft.Container):

    def init(self):
        self.state : Literal["load","camera","screen"] = "load"
        self.cap_num = ft.TextField(
            label=0, 
            value=0, keyboard_type=ft.KeyboardType.NUMBER,
            input_filter=ft.NumbersOnlyInputFilter(),
            width=50, 
            disabled=True
            )

        self.chbox_load = ft.Checkbox(value=False, on_change=self.toggleLoad)
        self.chbox_camera = ft.Checkbox(value=False, on_change=self.toggleCamera)
        self.chbox_screen =  ft.Checkbox(value=False, on_change=self.toggleScreen)
         
        self.content = ft.Column( controls = [ 
            ft.Text("Выберите источник изображения"), 
            ft.Row(controls=[
                self.chbox_load, 
                ft.Text("Загрузить изображение")
            ]), 
            ft.Row(controls=[
                self.chbox_camera, 
                ft.Text("Включить вебкамеру"),
                self.cap_num

            ]), 
            ft.Row(controls=[
                self.chbox_screen, 
                ft.Text("Снимок экрана")
            ]) 
                        ]
        )
        
        self.update_chboxes_states(); 



    def toggleLoad(self, e):
        self.state = "load"
        self.update_chboxes_states(); 
       
    def toggleCamera(self, e):
        self.state = "camera"
        self.update_chboxes_states(); 
    
    def toggleScreen(self, e):
        self.state = "screen"
        self.update_chboxes_states(); 

    def update_chboxes_states(self):
        self.chbox_load.value = (self.state == 'load')
        self.chbox_camera.value = (self.state == 'camera')
        self.chbox_screen.value = (self.state == 'screen')