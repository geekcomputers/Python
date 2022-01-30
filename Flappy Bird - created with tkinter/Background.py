from tkinter import Tk, Canvas

from PIL.Image import open as openImage
from PIL.ImageTk import PhotoImage


class Background(Canvas):
    """
    Classe para gerar um plano de fundo animado
    """

    __background = []
    __stop = False

    def __init__(self, tk_instance, *geometry, fp="background.png", animation_speed=50):

        # Verifica se o parâmetro tk_instance é uma instância de Tk
        if not isinstance(tk_instance, Tk):
            raise TypeError("The tk_instance argument must be an instance of Tk.")

        # Recebe o caminho de imagem e a velocidade da animação
        self.image_path = fp
        self.animation_speed = animation_speed

        # Recebe a largura e altura do widget
        self.__width = geometry[0]
        self.__height = geometry[1]

        # Inicializa o construtor da classe Canvas
        Canvas.__init__(
            self, master=tk_instance, width=self.__width, height=self.__height
        )

        # Carrega a imagem que será usada no plano de fundo
        self.__bg_image = self.getPhotoImage(
            image_path=self.image_path,
            width=self.__width,
            height=self.__height,
            closeAfter=True,
        )[0]

        # Cria uma imagem que será fixa, ou seja, que não fará parte da animação e serve em situações de bugs na animação
        self.__background_default = self.create_image(
            self.__width // 2, self.__height // 2, image=self.__bg_image
        )

        # Cria as imagens que serão utilizadas na animação do background
        self.__background.append(
            self.create_image(
                self.__width // 2, self.__height // 2, image=self.__bg_image
            )
        )
        self.__background.append(
            self.create_image(
                self.__width + (self.__width // 2),
                self.__height // 2,
                image=self.__bg_image,
            )
        )

    def getBackgroundID(self):
        """
        Retorna os id's das imagens de background
        """
        return [self.__background_default, *self.__background]

    @staticmethod
    def getPhotoImage(
        image=None, image_path=None, width=None, height=None, closeAfter=False
    ):
        """
        Retorna um objeto da classe PIL.ImageTk.PhotoImage de uma imagem e as imagens criadas de PIL.Image
        (photoImage, new, original)

        @param image: Instância de PIL.Image.open
        @param image_path: Diretório da imagem
        @param width: Largura da imagem
        @param height: Altura da imagem
        @param closeAfter: Se True, a imagem será fechada após ser criado um PhotoImage da mesma
        """

        if not image:
            if not image_path:
                return

            # Abre a imagem utilizando o caminho dela
            image = openImage(image_path)

        # Será redimesionada a imagem somente se existir um width ou height
        if not width:
            width = image.width
        if not height:
            height = image.height

        # Cria uma nova imagem já redimensionada
        newImage = image.resize([width, height])

        # Cria um photoImage
        photoImage = PhotoImage(newImage)

        # Se closeAfter for True, ele fecha as imagens
        if closeAfter:
            # Fecha a imagem nova
            newImage.close()
            newImage = None

            # Fecha a imagem original
            image.close()
            image = None

        # Retorna o PhotoImage da imagem,a nova imagem que foi utilizada e a imagem original
        return photoImage, newImage, image

    def reset(self):
        """
        Método para resetar o background, apagando todos os itens que não sejam o plano de fundo
        """

        # Deleta todos os itens do canvas
        self.delete("all")

        # Para a animação passando False para o atributo "stop"
        self.__stop = False

        # Limpa a lista de imagens usadas na animação
        self.__background.clear()

        # Cria uma imagem que será fixa, ou seja, que não fará parte da animação e serve em situações de bugs na animação
        self.__background_default = self.create_image(
            self.__width // 2, self.__height // 2, image=self.__bg_image
        )

        # Cria as imagens que serão utilizadas na animação do background
        self.__background.append(
            self.create_image(
                self.__width // 2, self.__height // 2, image=self.__bg_image
            )
        )
        self.__background.append(
            self.create_image(
                self.__width + (self.__width // 2),
                self.__height // 2,
                image=self.__bg_image,
            )
        )

    def run(self):
        """
        Método para iniciar a animação do background
        """

        # Enquanto o atributo "stop" for False, a animação continuará em um loop infinito
        if not self.__stop:

            # Move as imagens de background na posição X
            self.move(self.__background[0], -10, 0)
            self.move(self.__background[1], -10, 0)
            self.tag_lower(self.__background[0])
            self.tag_lower(self.__background[1])
            self.tag_lower(self.__background_default)

            # Se a primeira imagem da lista tiver saído da área do widget, uma nova será criada depois da segunda imagem
            if self.bbox(self.__background[0])[2] <= 0:
                # Deleta a primeira imagem da lista (imagem que saiu da área do widget)
                self.delete(self.__background[0])
                self.__background.remove(self.__background[0])

                # Cria uma nova imagem a partir da última imagem da animação
                width = self.bbox(self.__background[0])[2] + self.__width // 2
                self.__background.append(
                    self.create_image(width, self.__height // 2, image=self.__bg_image)
                )

            # Executa novamente o método depois de um certo tempo
            self.after(self.animation_speed, self.run)

    def stop(self):
        """
        Método para parar a animação do background
        """
        self.__stop = True
