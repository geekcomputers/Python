"""

wanna try some GUI based calculator, here is one!
Try it, brake it. If you find some bug and have better way ahead, i welcome your change :)

Install dependencies:
    pip install kivy==2.3.1 kivymd==1.1.1

"""

from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton  # Fixed F403
from kivy.lang import Builder

opt = ["(", ")", "X", "/", "+", "-"]
opt_check = ["X", "/", "+", "-"]

cal = """
MDScreen:
    MDBoxLayout:
        orientation:'vertical'
        MDLabel:
            text:'I welcome you!'
            adaptive_height:True
            halign:'center'

        MDTextField:
            id:field
            
            font_size:dp(60)
            pos_hint:{'top':1}
            size_hint_x:1
            size_hint_y:.4
            readonly:True
            multiline:True

        MDGridLayout:
            id:grid
            cols:4
"""


class calculator(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Dark"
        return Builder.load_string(cal)

    def on_start(self):

        self.root.ids.grid.add_widget(
            MDFlatButton(text="AC", on_release=self.delete_all, size_hint=[1, 1])
        )
        self.root.ids.grid.add_widget(
            MDFlatButton(text="del", on_release=self.delete, size_hint=[1, 1])
        )
        self.root.ids.grid.add_widget(
            MDFlatButton(text="^", on_release=self.to_field, size_hint=[1, 1])
        )

        for i in range(len(opt)):
            self.root.ids.grid.add_widget(
                MDFlatButton(
                    text=opt[i], on_release=self.to_field_opt, size_hint=[1, 1]
                )
            )

        for i in range(10):
            self.root.ids.grid.add_widget(
                MDFlatButton(text=str(i), on_release=self.to_field, size_hint=[1, 1])
            )

        self.root.ids.grid.add_widget(
            MDFlatButton(text="=", on_release=self.calculate, size_hint=[1, 1])
        )

    def to_field(self, btn):
        if self.root.ids.field.text == "undefined":
            self.root.ids.field.text = ""
        self.root.ids.field.text = self.root.ids.field.text + btn.text

    def to_field_opt(self, btn):
        if self.root.ids.field.text == "undefined":
            self.root.ids.field.text = ""

        elif btn.text != "(" and btn.text != ")" and self.root.ids.field.text == "":
            self.root.ids.field.text = f"0+{btn.text}"

        elif self.root.ids.field.text != "" and btn.text in opt_check:
            if self.root.ids.field.text[-1] in opt_check:
                self.root.ids.field.text = self.root.ids.field.text[:-1] + btn.text
            else:
                self.root.ids.field.text = self.root.ids.field.text + btn.text

        else:
            self.root.ids.field.text = self.root.ids.field.text + btn.text

    def delete_all(self, del_all_btn):
        self.root.ids.field.text = ""

    def delete(self, del_btn):
        self.root.ids.field.text = self.root.ids.field.text[:-1]

    def calculate(self, cal_btn):
        ch_opt_list = ["X", "^"]
        with_opt = ["*", "**"]
        raw = self.root.ids.field.text

        for opt in ch_opt_list:
            raw = raw.replace(opt, with_opt[ch_opt_list.index(opt)])

        try:
            self.root.ids.field.text = str(eval(raw))
        except Exception:  # Fixed E722
            self.root.ids.field.text = "undefined"


if __name__ == "__main__":
    calculator().run()
