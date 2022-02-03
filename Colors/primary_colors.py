def diff(a, b):
    """
    TODO: fix this function!!
    """
    return a - b


def simpleColor(r, g, b):
    """simpleColor obtiene el nombre del color mas general al cual se acerca su formato R G B"""
    r = int(r)
    g = int(g)
    b = int(b)
    bg = ir = 0  # TODO: Fix these variables
    try:
        # ROJO --------------------------------------------------
        if r > g and r > b:

            rg = diff(r, g)  # distancia rojo a verde
            rb = diff(r, b)  # distancia rojo a azul

            if g < 65 and b < 65 and rg > 60:  # azul y verde sin luz
                return "ROJO"

            gb = diff(g, b)  # distancia de verde a azul

            if rg < rb:  # Verde mayor que Azul
                if gb < rg:  # Verde mas cerca de Azul
                    if gb >= 30 and rg >= 80:
                        return "NARANJA"
                    elif gb <= 20 and rg >= 80:
                        return "ROJO"
                    elif gb <= 20 and b > 175:
                        return "CREMA"

                    else:
                        return "CHOCOLATE"
                else:  # Verde mas cerca de Rojo
                    if rg > 60:
                        return "NARANJA*"
                    elif r > 125:
                        return "AMARILLO"
                    else:
                        return "COCHOLATE"
            elif rg > rb:  # Azul mayor que verde
                if bg < rb:  # Verde mas cerca de Azul
                    if gb < 60:
                        if r > 150:
                            return "ROJO 2"
                        else:
                            return "MARRON"
                    elif g > 125:
                        return "ROSADO"
                    else:
                        return "ROJO 3"
                else:  # Verde mas cerca de Rojo
                    if rb < 60:
                        if r > 160:
                            return "ROSADO*"
                        else:
                            return "ROJO"
                    else:
                        return "ROJO"

            else:  # g y b iguales
                if rg > 20:
                    if r >= 100 and b < 60:
                        return "ROJO"
                    elif r >= 100:
                        return "ROJO"
                    else:
                        return "MARRON"

                else:
                    return "GRIS"
        # VERDE ---------------------------------------------------
        elif g > r and g > b:
            gb = diff(g, b)  # distancia verde a azul
            gr = diff(g, r)  # distancia verde a rojo

            if r < 65 and b < 65 and gb > 60:  # rojo y azul sin luz
                return "VERDE"

            rb = diff(r, b)  # distancia de rojo a azul

            if r > b:  # ROJO > AZUL
                if gr < gb:  # Verde con Rojo

                    if rb >= 150 and gr <= 20:
                        return "AMARILLO"
                    else:
                        return "VERDE"
                else:  # ...Verde
                    return "VERDE"

            elif r < b:  # AZUL > ROJO
                if gb < gr:  # Verde con Azul

                    if gb <= 20:
                        return "TURQUESA"
                    else:
                        return "VERDE"
                else:  # ...Verde
                    return "VERDE"

            else:  # r y b iguales
                if gb > 10:
                    return "VERDE"
                else:
                    return "GRIS"

        # AZUL ------------------------------------------------------
        elif b > r and b > g:
            bg = diff(b, g)  # distancia azul a verde
            br = diff(b, r)  # distancia azul a rojo

            if r < 65 and g < 65 and bg > 60:  # rojo y verde sin luz
                return "AZUL"

            rg = diff(r, g)  # distancia de rojo a verde

            if g < r:  # ROJO  > VERDE
                if bg < rg:  # Azul con Verde
                    if bg <= 20:
                        return "TURQUESA"
                    else:
                        return "CELESTE"
                else:  # ...Azul
                    if rg <= 20:
                        if r >= 150:
                            return "LILA"
                        else:
                            return "AZUL *************"
                    else:
                        return "AZUL"

            elif g > r:  #  VERDE > ROJO
                if br < rg:  # Azul con rojo
                    if br <= 20:
                        if r > 150 and g < 75:
                            return "ROSADO FIUSHA"
                        elif ir > 150:
                            return "LILA"
                        else:
                            return "MORADO"
                    else:
                        return "MORADO"

                else:  # ...Azul
                    if rg <= 20:
                        if bg <= 20:
                            return "GRIS"
                        else:
                            return "AZUL"
            else:  # r y g iguales
                if bg > 20:
                    if r >= 100 and b < 60:
                        return "ROJO"
                    elif r >= 100:
                        return "ROJO"
                    else:
                        return "MARRON"
                else:
                    return "GRIS"

        # IGUALES---------------------------------------
        else:
            return "GRIS"

    except:

        return "Not Color"


# ---------------------------------------------------------------------------------------------------
# Puedes probar asi: python primary_colors.py 120,0,0   , esto resultara en un ROJO como respuesta
# --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print(simpleColor(sys.argv[1], sys.argv[2], sys.argv[3]))
