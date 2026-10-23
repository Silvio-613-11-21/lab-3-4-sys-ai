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
        files = await pick_image()
        # отмена диалога больше не роняет программу на files[0]
        if not files:
            self.selected_file_1.value = "Cancelled!"
            self.update()
            return
        self.selected_file_1.value = files.name
        self.img_src_1 = files.path
        print(self.img_src_1)
        self.update()

    async def handle_pick_file_2(self, e: ft.Event[ft.Button]):
        files = await pick_image()
        if not files:
            self.selected_file_2.value = "Cancelled!"
            self.update()
            return
        self.selected_file_2.value = files.name
        self.img_src_2 = files.path
        print(self.img_src_2)
        self.update()


async def pick_image():
    """Диалог выбора изображения. Возвращает файл или None (отмена)."""
    files = await ft.FilePicker().pick_files(
        dialog_title="Выберите изображение",
        file_type=ft.FilePickerFileType.CUSTOM,
        allowed_extensions=["jpg", "jpeg", "png", "gif", "bmp"],
    )
    if not files:
        return None
    return files[0]