import flet as ft

@ft.control
class CVWindow(ft.Container):
    def init(self):

        self.pic = ft.Image(
            src="",
            # fit=ft.BoxFit.CONTAIN,
           
            height=500,
            width=900
        )

        self.txt = ft.TextField(value='',
            
            # height=70,
            width= 900,
            multiline=True,
             
                                ) 

        self.content = ft.Column(
            controls= [
                self.pic, 
                self.txt
            ], 
            
            alignment=ft.MainAxisAlignment.END,          
            horizontal_alignment=ft.CrossAxisAlignment.START, 
            spacing=10,
            scroll=ft.ScrollMode.AUTO      
        )

        
        self.expand = True
        self.width = 900
        self.height = 600
        

    def set_src(self, src: str):
        self.pic.src = src
        self.update()

