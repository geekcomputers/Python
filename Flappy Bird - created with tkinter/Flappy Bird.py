__author__ = "Jean Loui Bernard Silva de Jesus"
__version__ = "1.0"

import os.path
from datetime import timedelta
from time import time
from tkinter import Tk, Button

from Background import Background
from Bird import Bird
from Settings import Settings
from Tubes import Tubes


class App(Tk, Settings):
    """
    Classe principal do jogo onde tudo será executado
    """

    # Variáveis privadas e ajustes internos
    __background_animation_speed = 720
    __bestScore = 0
    __bird_descend_speed = 38.4
    __buttons = []
    __playing = False
    __score = 0
    __time = "%H:%M:%S"

    def __init__(self):

        Tk.__init__(self)
        self.setOptions()

        # Se o tamanho da largura e altura da janela forem definidos, eles serão usados no jogo.
        # Caso eles tenham o valor None, o tamanho da janela será o tamanho do monitor do usuário.

        if all([self.window_width, self.window_height]):
            self.__width = self.window_width
            self.__height = self.window_height
        else:
            self.__width = self.winfo_screenwidth()
            self.__height = self.winfo_screenheight()

        # Configura a janela do programa
        self.title(self.window_name)
        self.geometry("{}x{}".format(self.__width, self.__height))
        self.resizable(*self.window_rz)
        self.attributes("-fullscreen", self.window_fullscreen)
        self["bg"] = "black"

        # Verifica se existem as imagens do jogo
        for file in self.images_fp:
            if not os.path.exists(file):
                raise FileNotFoundError(
                    "The following file was not found:\n{}".format(file)
                )

        # Carrega a imagem do botão para começar o jogo
        self.__startButton_image = Background.getPhotoImage(
            image_path=self.startButton_fp,
            width=(self.__width // 100) * self.button_width,
            height=(self.__height // 100) * self.button_height,
            closeAfter=True,
        )[0]

        # Carrega a imagem do botão para sair do jogo
        self.__exitButton_image = Background.getPhotoImage(
            image_path=self.exitButton_fp,
            width=(self.__width // 100) * self.button_width,
            height=(self.__height // 100) * self.button_height,
            closeAfter=True,
        )[0]

        # Carrega a imagem do título do jogo
        self.__title_image = Background.getPhotoImage(
            image_path=self.title_fp,
            width=(self.__width // 100) * self.title_width,
            height=(self.__height // 100) * self.title_height,
            closeAfter=True,
        )[0]

        # Carrega a imagem do placar do jogo
        self.__scoreboard_image = Background.getPhotoImage(
            image_path=self.scoreboard_fp,
            width=(self.__width // 100) * self.scoreboard_width,
            height=(self.__height // 100) * self.scoreboard_height,
            closeAfter=True,
        )[0]

        # Define a velocidade da animação do background com base na largura da janela
        self.__background_animation_speed //= self.__width / 100
        self.__background_animation_speed = int(self.__background_animation_speed)

        # Define a velocidade de descida do pássaro com base na altura da janela
        self.__bird_descend_speed //= self.__height / 100
        self.__bird_descend_speed = int(self.__bird_descend_speed)

    def changeFullscreenOption(self, event=None):
        """
        Método para colocar o jogo no modo "fullscreen" ou "window"
        """

        self.window_fullscreen = not self.window_fullscreen
        self.attributes("-fullscreen", self.window_fullscreen)

    def close(self, event=None):
        """
        Método para fechar o jogo
        """

        # Salva a melhor pontuação do jogador antes de sair do jogo
        self.saveScore()

        # Tenta interromper os processos
        try:
            self.__background.stop()
            self.__bird.kill()
            self.__tubes.stop()
        finally:
            quit()

    def createMenuButtons(self):
        """
        Método para criar os botões de menu
        """

        # Define o tamanho do botão em porcentagem com base no tamanho da janela
        width = (self.__width // 100) * self.button_width
        height = (self.__height // 100) * self.button_height

        # Cria um botão para começar o jogo
        startButton = Button(
            self,
            image=self.__startButton_image,
            bd=0,
            command=self.start,
            cursor=self.button_cursor,
            bg=self.button_bg,
            activebackground=self.button_activebackground,
        )
        # Coloca o botão dentro do background ( Canvas )
        self.__buttons.append(
            self.__background.create_window(
                (self.__width // 2) - width // 1.5,
                int(self.__height / 100 * self.button_position_y),
                window=startButton,
            )
        )

        # Cria um botão para sair do jogo
        exitButton = Button(
            self,
            image=self.__exitButton_image,
            bd=0,
            command=self.close,
            cursor=self.button_cursor,
            bg=self.button_bg,
            activebackground=self.button_activebackground,
        )

        # Coloca o botão dentro do background ( Canvas )
        self.__buttons.append(
            self.__background.create_window(
                (self.__width // 2) + width // 1.5,
                int(self.__height / 100 * self.button_position_y),
                window=exitButton,
            )
        )

    def createScoreBoard(self):
        """
        Método para criar a imagem do placar do jogo no background
        junto com as informações do jogador.
        """

        # Define a posição X e Y
        x = self.__width // 2
        y = (self.__height // 100) * self.scoreboard_position_y

        # Calcula o tamanho da imagem do placar
        scoreboard_w = (self.__width // 100) * self.scoreboard_width
        scoreboard_h = (self.__width // 100) * self.scoreboard_height

        # Calcula a posição X e Y do texto da pontuação do último jogo
        score_x = x - scoreboard_w / 100 * 60 / 2
        score_y = y + scoreboard_h / 100 * 10 / 2

        # Calcula a posição X e Y do texto da melhor pontuação do jogador
        bestScore_x = x + scoreboard_w / 100 * 35 / 2
        bestScore_y = y + scoreboard_h / 100 * 10 / 2

        # Calcula a posição X e Y do texto do tempo de jogo
        time_x = x
        time_y = y + scoreboard_h / 100 * 35 / 2

        # Define a fonte dos textos
        font = (self.text_font, int(0.02196 * self.__width + 0.5))

        # Cria a imagem do placar no background
        self.__background.create_image(x, y, image=self.__scoreboard_image)

        # Cria texto para mostrar o score do último jogo
        self.__background.create_text(
            score_x,
            score_y,
            text="Score: %s" % self.__score,
            fill=self.text_fill,
            font=font,
        )

        # Cria texto para mostrar a melhor pontuação do jogador
        self.__background.create_text(
            bestScore_x,
            bestScore_y,
            text="Best Score: %s" % self.__bestScore,
            fill=self.text_fill,
            font=font,
        )

        # Cria texto para mostrar o tempo de jogo
        self.__background.create_text(
            time_x,
            time_y,
            text="Time: %s" % self.__time,
            fill=self.text_fill,
            font=font,
        )

    def createTitleImage(self):
        """
        Método para criar a imagem do título do jogo no background
        """

        self.__background.create_image(
            self.__width // 2,
            (self.__height // 100) * self.title_position_y,
            image=self.__title_image,
        )

    def deleteMenuButtons(self):
        """
        Método para deletar os botões de menu
        """

        # Deleta cada botão criado dentro do background
        for item in self.__buttons:
            self.__background.delete(item)

        # Limpa a lista de botões
        self.__buttons.clear()

    def gameOver(self):
        """
        Método de fim de jogo
        """

        # Calcula o tempo jogado em segundos e depois o formata
        self.__time = int(time() - self.__time)
        self.__time = str(timedelta(seconds=self.__time))

        # Interrompe a animação do plano de fundo e a animação dos tubos
        self.__background.stop()
        self.__tubes.stop()

        # Declara que o jogo não está mais em execução
        self.__playing = False

        # Cria os botões inciais
        self.createMenuButtons()

        # Cria image do título do jogo
        self.createTitleImage()

        # Cria imagem do placar e mostra as informações do jogo passado
        self.createScoreBoard()

    def increaseScore(self):
        """
        Método para aumentar a pontuação do jogo atual do jogador
        """

        self.__score += 1
        if self.__score > self.__bestScore:
            self.__bestScore = self.__score

    def init(self):
        """
        Método para iniciar o programa em si, criando toda a parte gráfica inicial do jogo
        """

        # self.createMenuButtons()
        self.loadScore()

        # Cria o plano de fundo do jogo
        self.__background = Background(
            self,
            self.__width,
            self.__height,
            fp=self.background_fp,
            animation_speed=self.__background_animation_speed,
        )

        # Foca o plano de fundo para que seja possível definir os eventos
        self.__background.focus_force()
        # Define evento para trocar o modo de janela para "fullscreen" ou "window"
        self.__background.bind(
            self.window_fullscreen_event, self.changeFullscreenOption
        )
        # Define evento para começar o jogo
        self.__background.bind(self.window_start_event, self.start)
        # Define evento para sair do jogo
        self.__background.bind(self.window_exit_event, self.close)

        # Define um método caso o usuário feche a janela do jogo
        self.protocol("WM_DELETE_WINDOW", self.close)

        # Empacota o objeto background
        self.__background.pack()

        # Cria os botões do menu do jogo
        self.createMenuButtons()

        # Cria imagem do título do jogo
        self.createTitleImage()

        # Cria um pássaro inicial no jogo
        self.__bird = Bird(
            self.__background,
            self.gameOver,
            self.__width,
            self.__height,
            fp=self.bird_fp,
            event=self.bird_event,
            descend_speed=self.__bird_descend_speed,
        )

    def loadScore(self):
        """
        Método para carregar a pontuação do jogador
        """

        # Tenta carregar o placar do usuário
        try:
            file = open(self.score_fp)
            self.__bestScore = int(file.read(), 2)
            file.close()

        # Se não for possível, será criado um arquivo para guardar o placar
        except BaseException:
            file = open(self.score_fp, "w")
            file.write(bin(self.__bestScore))
            file.close()

    def saveScore(self):
        """
        Método para salvar a pontuação do jogador
        """

        with open(self.score_fp, "w") as file:
            file.write(bin(self.__bestScore))

    def start(self, event=None):
        """
        Método para inicializar o jogo
        """

        # Este método é executado somente se o jogador não estiver já jogando
        if self.__playing:
            return

        # Reinicia o placar
        self.__score = 0
        self.__time = time()

        # Remove os botões de menu
        self.deleteMenuButtons()

        # Reinicia o background
        self.__background.reset()

        # Inicializa a animação do background se True
        if self.background_animation:
            self.__background.run()

        # Cria um pássaro no jogo
        self.__bird = Bird(
            self.__background,
            self.gameOver,
            self.__width,
            self.__height,
            fp=self.bird_fp,
            event=self.bird_event,
            descend_speed=self.__bird_descend_speed,
        )

        # Cria tubos no jogo
        self.__tubes = Tubes(
            self.__background,
            self.__bird,
            self.increaseScore,
            self.__width,
            self.__height,
            fp=self.tube_fp,
            animation_speed=self.__background_animation_speed,
        )

        # Inicializa a animação do pássaro e dos tubos
        self.__bird.start()
        self.__tubes.start()


if __name__ == "__main__":
    try:
        app = App()
        app.init()
        app.mainloop()

    except FileNotFoundError as error:
        print(error)
