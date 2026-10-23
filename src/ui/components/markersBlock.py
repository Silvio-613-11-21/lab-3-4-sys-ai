import flet as ft


@ft.control
class MarkersBlock(ft.Container):
    

    def init(self):
        self.show_markers = ft.Checkbox(value=True, )
        self.unit_markers = ft.Checkbox(value=True, disabled=True)
  
         
        self.content = ft.Column( controls = [ 
            ft.Row(controls=[
                self.show_markers, 
                ft.Text("Отобразить маркеры")
            ]), 
            ft.Row(controls=[
                self.unit_markers, 
                ft.Text("Соединить подобные маркеры")
            ])              
            ]
        )
        

        
    
