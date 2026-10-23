import flet as ft
from .components.imgSourceBlock import ImgSourceBlock
from .components.loadImgsBlock import LoadImgsBlock
from .components.markersBlock import MarkersBlock
from .components.cvWindow import CVWindow

@ft.control
class UI(ft.Row):
    def init(self):
       
        self.img_source = ImgSourceBlock() 
        self.imgs_loads = LoadImgsBlock()
        self.markers_state = MarkersBlock()
        self.cv_window = CVWindow()
        
        self.controls = [
            ft.Column(controls=[
                self.img_source, 
                self.imgs_loads,
                self.markers_state,
            ] , 
                spacing = 20,
                expand=True, 
                ),

            self.cv_window
        ]

        
        

        addStyle_(self.img_source)
        addStyle_(self.imgs_loads)
        addStyle_(self.markers_state)

        addStyle(self.cv_window.pic)
        addStyle(self.cv_window.txt)
        # addStyle(self.cv_window)

def addStyle_(comp: ft.Container):
    addStyle(comp)
    comp.width = 300

def addStyle(comp: ft.Container):
    comp.padding = 10
            # self.border_radius = 10
    comp.border = ft.Border.all(1, ft.Colors.WHITE)
    comp.shadow = ft.BoxShadow(
                                    spread_radius=1,
                                    blur_radius=15,
                                    color=ft.Colors.GREY_600,
                                    offset=ft.Offset(5, 5),
                                )
    comp.border_radius = 4
    comp.bgcolor = ft.Colors.BLACK

       
