import flet as ft

@ft.control
class LoadImgsBlock(ft.Container):    

    img_src_1: str | None = None 
    img_src_2: str | None = None

    def init(self ):
        self.selected_file_1 = ft.Text()
        self.selected_file_2 = ft.Text()
        self.text_check = ft.Checkbox(value=False)

        self.content = ft.Column(controls=[
            ft.Text("Выберите изображение сцены:"),
            ft.Button(
                content="Выберите файл",
                icon = ft.Icons.UPLOAD_FILE,
                on_click=self.handle_pick_file_1
            ),
            self.selected_file_1, 

            ft.Text("Выберите изображение объекта:"),
            ft.Button(
                content="Выберите файл",
                icon = ft.Icons.UPLOAD_FILE,
                on_click=self.handle_pick_file_2
                ),
            self.selected_file_2,    

            ft.Row(
                controls= [
                    self.text_check, 
                    ft.Text("Распознать текст")
                ]
            )
        ])
        
    
    async def handle_pick_file_1(self, e: ft.Event[ft.Button]):
        files = await ft.FilePicker().pick_files(allowed_extensions=["jpg", "jpeg", "png", "gif"])
        self.selected_file_1.value = (
            files[0].name if files else "Cancelled!"
        )
        self.img_src_1 =  files[0].path
        print(self.img_src_1)

    async def handle_pick_file_2(self, e: ft.Event[ft.Button]):
        files = await ft.FilePicker().pick_files(allowed_extensions=["jpg", "jpeg", "png",  "gif"])
        self.selected_file_2.value = (
            files[0].name if files else "Cancelled!"
        )
        self.img_src_2 =  files[0].path
        print(self.img_src_2)