from random import randint
from threading import Thread

from Background import Background
from Bird import Bird
from PIL.Image import open as openImage
from PIL.ImageTk import PhotoImage


class Tubes(Thread):
    """
    Classe para criar tubos
    """

    __distance = 0
    __move = 10
    __pastTubes = []

    def __init__(
        self,
        background,
        bird,
        score_function=None,
        *screen_geometry,
        fp=("tube.png", "tube_mourth"),
        animation_speed=50
    ):

        # Verifica os parâmetros passados e lança um erro caso algo esteja incorreto
        if not isinstance(background, Background):
            raise TypeError(
                "The background argument must be an instance of Background."
            )
        if not len(fp) == 2:
            raise TypeError(
                "The parameter fp should be a sequence containing the path of the images of the tube body and the tube mouth."
            )
        if not isinstance(bird, Bird):
            raise TypeError("The birdargument must be an instance of Bird.")
        if not callable(score_function):
            raise TypeError("The score_function argument must be a callable object.")

        Thread.__init__(self)

        # Instância os parâmetros
        self.__background = background
        self.image_path = fp
        self.__animation_speed = animation_speed
        self.__score_method = score_function

        # Recebe a largura e altura do background
        self.__width = screen_geometry[0]
        self.__height = screen_geometry[1]

        # Recebe o tamanho do pássaro
        self.__bird_w = bird.width
        self.__bird_h = bird.height

        # Calcula a largura e altura da imagem
        self.__imageWidth = (self.__width // 100) * 10
        self.__imageHeight = (self.__height // 100) * 5

        # Cria uma lista para guardar imagens dos tubos
        try:
            self.deleteAll()
        except BaseException:
            self.__background.tubeImages = []

        # Cria uma lista somente para guardar as imagens futuras dos corpos dos tubos gerados
        self.__background.tubeImages.append([])

        # Carrega a imagem da boca do tubo
        self.__background.tubeImages.append(
            self.getPhotoImage(
                image_path=self.image_path[1],
                width=self.__imageWidth,
                height=self.__imageHeight,
                closeAfter=True,
            )[0]
        )

        # Carrega imagem do corpo do tubo
        self.__background.tubeImages.append(
            self.getPhotoImage(
                image_path=self.image_path[0],
                width=self.__imageWidth,
                height=self.__imageHeight,
            )[1]
        )

        # Calcula a distância mínima inicial entre os tubos
        self.__minDistance = int(self.__imageWidth * 4.5)

        self.__stop = False
        self.__tubes = []

    def createNewTubes(self):
        """
        Método para criar 2 novos tubos (baixo e cima) numa mesma posição X
        """

        # Cria uma lista para armazenar as partes do corpo do tubo de cima
        tube1 = []

        # Define a posição X que o tubo de cima aparecerá inicialmente no background
        width = self.__width + (self.__imageWidth)

        # Define uma posição Y para o tubo aleatóriamente respeitando algumas regras que são:
        # Espaço para o pássaro passar e espaço para adicionar o tubo de baixo.

        height = randint(
            self.__imageHeight // 2,
            self.__height - (self.__bird_h * 2) - self.__imageHeight,
        )

        # Cria e adiciona à lista do corpo do tubo de cima, a boca do tubo
        tube1.append(
            self.__background.create_image(
                width, height, image=self.__background.tubeImages[1]
            )
        )

        # Cria uma nova imagem na lista de imagens com a altura sendo igual a posição Y do tubo de cima
        self.__background.tubeImages[0].append(
            [
                self.getPhotoImage(
                    image=self.__background.tubeImages[2],
                    width=self.__imageWidth,
                    height=height,
                )[0],
            ]
        )

        # Define a posição Y do corpo do tubo de cima
        y = (height // 2) + 1 - (self.__imageHeight // 2)

        # Cria e adiciona à lista do corpo do tubo de cima, o corpo do tubo
        tube1.append(
            self.__background.create_image(
                width, y, image=self.__background.tubeImages[0][-1][0]
            )
        )

        ###############################################################################################################
        ###############################################################################################################

        # Cria uma lista para armazenar as partes do corpo do tubo de baixo
        tube2 = []

        # A posição Y do tubo de baixo é calculada com base na posição do tubo de cima, mais o tamanho do pássaro
        height = height + (self.__bird_h * 2) + self.__imageHeight - 1

        # Cria e adiciona à lista do corpo do tubo de baixo, a boca do tubo
        tube2.append(
            self.__background.create_image(
                width, height, image=self.__background.tubeImages[1]
            )
        )

        # Define a altura da imagem do corpo do tubo de baixo
        height = self.__height - height

        # Cria uma nova imagem na lista de imagens com a altura sendo igual a posição Y do tubo de baixo
        self.__background.tubeImages[0][-1].append(
            self.getPhotoImage(
                image=self.__background.tubeImages[2],
                width=self.__imageWidth,
                height=height,
            )[0]
        )

        # Define a posição Y do corpo do tubo de baixo
        y = (self.__height - (height // 2)) + self.__imageHeight // 2

        # Cria e adiciona à lista do corpo do tubo de baixo, o corpo do tubo
        tube2.append(
            self.__background.create_image(
                width, y, image=self.__background.tubeImages[0][-1][1]
            )
        )

        # Adiciona à lista de tubos os tubos de cima e de baixo da posição X
        self.__tubes.append([tube1, tube2])

        # Define a distância como sendo ZERO
        self.__distance = 0

    def deleteAll(self):
        """
        Método para deletar todos os tubos gerados
        """

        # Deleta os tubos gerados no background
        for tubes in self.__tubes:
            for tube in tubes:
                for body in tube:
                    self.__background.delete(body)

        self.__background.clear()
        self.__background.tubeImages.clear()

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

    def move(self):
        """
        Método para mover todos os tubos
        """

        # Cria uma variável auxilar para checar se o método de pontuar foi executado
        scored = False

        # Move os tubos gerados no background
        for tubes in self.__tubes:
            for tube in tubes:

                # Verifica se o pássaro passou do tubo. Caso sim, o método para pontuar será executado
                if not scored:

                    # Recebe a posição do cano
                    x2 = self.__background.bbox(tube[0])[2]

                    # Se a posição "x2" do tubo for menor que a posição "x1" do pássaro e se ainda não tiver sido
                    # pontuado este mesmo cano, o método para pontuar será chamado.

                    if (self.__width / 2) - (self.__bird_w / 2) - self.__move < x2:
                        if x2 <= (self.__width / 2) - (self.__bird_w / 2):

                            # Verifica se o tubo está na lista de tubos passados
                            if not tube[0] in self.__pastTubes:
                                # Chama o método para pontuar e adiciona o tubo pontuado à lista de tubos passados
                                self.__score_method()
                                self.__pastTubes.append(tube[0])
                                scored = True

                # Move cada parte do copo do tubo no background
                for body in tube:
                    self.__background.move(body, -self.__move, 0)

    def run(self):
        """
        Método para gerar os tubos no background e fazer a sua animação em um loop infinito
        """

        # Se o método "stop" tiver sido chamado, a animação será encerrada
        if self.__stop:
            return

        # Se os tubos ( cima e baixo ) de uma posição X tiverem sumido da área do background,
        # eles serão apagados juntamente com suas imagens e todos os seus dados.

        if (
            len(self.__tubes) >= 1
            and self.__background.bbox(self.__tubes[0][0][0])[2] <= 0
        ):

            # Apaga todo o corpo do tubo dentro do background
            for tube in self.__tubes[0]:
                for body in tube:
                    self.__background.delete(body)

            # Remove os tubos ( cima e baixo ) da lista de tubos
            self.__background.tubeImages[0].remove(self.__background.tubeImages[0][0])

            # Remove a imagem do corpo do tubo da lista de imagens
            self.__tubes.remove(self.__tubes[0])

            # Remove o primeiro objeto da lista de tubos passados
            self.__pastTubes.remove(self.__pastTubes[0])

        # Se a distancia entre o último tubo criado e o lado "x2" do background for maior que a distância
        # mínima estabelecida, então um novo tubo será criado.

        if self.__distance >= self.__minDistance:
            self.createNewTubes()
        else:
            # Aumenta a distancia conforme os tubos se movem
            self.__distance += self.__move

        # Move os tubos
        self.move()

        # Executa novamente o método em um determinado tempo
        self.__background.after(self.__animation_speed, self.run)

    def stop(self):
        """
        Método para interromper a Thread
        """

        self.__stop = True
