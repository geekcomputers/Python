from threading import Thread

from Background import Background
from PIL.Image import open as openImage
from PIL.ImageTk import PhotoImage


class Bird(Thread):
    """
    Classe para criar um pássaro
    """

    __tag = "Bird"
    __isAlive = None
    __going_up = False
    __going_down = 0
    __times_skipped = 0
    __running = False

    decends = 0.00390625
    climbsUp = 0.0911458333

    def __init__(self, background, gameover_function, *screen_geometry, fp="bird.png", event="<Up>", descend_speed=5):

        # Verifica se "background" é uma instância de Background e se o "gamerover_method" é chamável

        if not isinstance(background, Background): raise TypeError(
            "The background argument must be an instance of Background.")
        if not callable(gameover_function): raise TypeError("The gameover_method argument must be a callable object.")

        # Instância os parâmetros
        self.__canvas = background
        self.image_path = fp
        self.__descend_speed = descend_speed
        self.gameover_method = gameover_function

        # Recebe a largura e altura do background
        self.__width = screen_geometry[0]
        self.__height = screen_geometry[1]

        # Define a decida e subida do pássaro com base na altura do background
        self.decends *= self.__height
        self.decends = int(self.decends + 0.5)
        self.climbsUp *= self.__height
        self.climbsUp = int(self.climbsUp + 0.5)

        # Invoca o método construtor de Thread
        Thread.__init__(self)

        # Calcula o tamanho do pássaro com base na largura e altura da janela
        self.width = (self.__width // 100) * 6
        self.height = (self.__height // 100) * 11

        # Carrega e cria a imagem do pássaro no background
        self.__canvas.bird_image = \
        self.getPhotoImage(image_path=self.image_path, width=self.width, height=self.height, closeAfter=True)[0]
        self.__birdID = self.__canvas.create_image(self.__width // 2, self.__height // 2,
                                                   image=self.__canvas.bird_image, tag=self.__tag)

        # Define evento para fazer o pássaro subir
        self.__canvas.focus_force()
        self.__canvas.bind(event, self.jumps)
        self.__isAlive = True

    def birdIsAlive(self):
        """
        Método para verificar se o pássaro está vivo
        """

        return self.__isAlive

    def checkCollision(self):
        """
        Método para verificar se o pássaro ultrapassou a borda da janela ou colidiu com algo
        """

        # Recebe a posição do pássaro no background
        position = list(self.__canvas.bbox(self.__tag))

        # Se o pássaro tiver ultrapassado a borda de baixo do background, ele será declarado morto
        if position[3] >= self.__height + 20:
            self.__isAlive = False

        # Se o pássaro tiver ultrapassado a borda de cima do background, ele será declarado morto    
        if position[1] <= -20:
            self.__isAlive = False

        # Dá uma margem de erro ao pássaro de X pixels 
        position[0] += int(25 / 78 * self.width)
        position[1] += int(25 / 77 * self.height)
        position[2] -= int(20 / 78 * self.width)
        position[3] -= int(10 / 77 * self.width)

        # Define os objetos a serem ignorados em colisões
        ignored_collisions = self.__canvas.getBackgroundID()
        ignored_collisions.append(self.__birdID)

        # Verifica possíveis colisões com o pássaro
        possible_collisions = list(self.__canvas.find_overlapping(*position))

        # Remove das possíveis colisões os objetos ignorados
        for _id in ignored_collisions:
            try:
                possible_collisions.remove(_id)
            except:
                continue

        # Se houver alguma colisão o pássaro morre
        if len(possible_collisions) >= 1:
            self.__isAlive = False

        return not self.__isAlive

    def getTag(self):
        """
        Método para retornar a tag do pássaro
        """

        return self.__tag

    @staticmethod
    def getPhotoImage(image=None, image_path=None, width=None, height=None, closeAfter=False):
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
            if not image_path: return

            # Abre a imagem utilizando o caminho dela
            image = openImage(image_path)

        # Será redimesionada a imagem somente se existir um width ou height
        if not width: width = image.width
        if not height: height = image.height

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

    def jumps(self, event=None):
        """
        Método para fazer o pássaro pular
        """

        # Verifica se o pássaro saiu da área do background
        self.checkCollision()

        # Se o pássaro estiver morto, esse método não pode ser executado
        if not self.__isAlive or not self.__running:
            self.__going_up = False
            return

        # Declara que o pássaro está subindo
        self.__going_up = True
        self.__going_down = 0

        # Move o pássaro enquanto o limite de subida por animação não tiver excedido
        if self.__times_skipped < self.climbsUp:

            # Move o pássaro para cima
            self.__canvas.move(self.__tag, 0, -1)
            self.__times_skipped += 1

            # Executa o método novamente
            self.__canvas.after(3, self.jumps)

        else:

            # Declara que o pássaro não está mais subindo
            self.__going_up = False
            self.__times_skipped = 0

    def kill(self):
        """
        Método para matar o pássaro
        """

        self.__isAlive = False

    def run(self):
        """
        #Método para iniciar a animação do passáro caindo
        """

        self.__running = True

        # Verifica se o pássaro saiu da área do background
        self.checkCollision()

        # Enquanto o pássaro não tiver chegado em sua velocidade máxima, a velocidade aumentará em 0.05
        if self.__going_down < self.decends:
            self.__going_down += 0.05

        # Executa a animação de descida somente se o pássaro estiver vivo
        if self.__isAlive:

            # Executa a animação de descida somente se o pássaro não estiver subindo
            if not self.__going_up:
                # Move o pássaro para baixo
                self.__canvas.move(self.__tag, 0, self.__going_down)

            # Executa novamente o método
            self.__canvas.after(self.__descend_speed, self.run)

        # Se o pássaro estiver morto, será executado um método de fim de jogo
        else:
            self.__running = False
            self.gameover_method()
