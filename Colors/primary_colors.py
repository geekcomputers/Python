def diff(a, b):
    """
    Calculate the absolute difference between two values.
    This helps in determining the variance between color channels.
    
    Args:
        a (int/float): First value
        b (int/float): Second value
        
    Returns:
        int/float: Absolute difference between a and b
    """
    return abs(a - b)  # Fixed to return absolute difference (critical for color comparison)


def simpleColor(r, g, b):
    """
    Determines the general color name a given RGB value approximates to.
    Classification is based on comparing the intensity of red, green, and blue channels,
    as well as their mutual differences.
    
    Args:
        r (int/float): Red channel value (0-255)
        g (int/float): Green channel value (0-255)
        b (int/float): Blue channel value (0-255)
        
    Returns:
        str: General color name (e.g., "ROJO", "VERDE") or error message if invalid
    """
    try:
        # Convert inputs to integers and validate range
        r = int(r)
        g = int(g)
        b = int(b)
        
        if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
            return "Error: RGB values must be between 0 and 255"

        # RED DOMINANT --------------------------------------------------
        if r > g and r > b:
            red_green_diff = diff(r, g)  # Difference between red and green
            red_blue_diff = diff(r, b)   # Difference between red and blue

            # Pure red (green and blue are very low)
            if g < 65 and b < 65 and red_green_diff > 60:
                return "ROJO"

            green_blue_diff = diff(g, b)  # Difference between green and blue

            # Green is more prominent than blue
            if red_green_diff < red_blue_diff:
                if green_blue_diff < red_green_diff:  # Green closer to blue
                    if green_blue_diff >= 30 and red_green_diff >= 80:
                        return "NARANJA"
                    elif green_blue_diff <= 20 and red_green_diff >= 80:
                        return "ROJO"
                    elif green_blue_diff <= 20 and b > 175:
                        return "CREMA"
                    else:
                        return "CHOCOLATE"
                else:  # Green closer to red
                    if red_green_diff > 60:
                        return "NARANJA*"
                    elif r > 125:
                        return "AMARILLO"
                    else:
                        return "CHOCOLATE"  # Fixed typo from "COCHOLATE"
            
            # Blue is more prominent than green
            elif red_green_diff > red_blue_diff:
                if green_blue_diff < red_blue_diff:  # Green closer to blue
                    if green_blue_diff < 60:
                        return "ROJO 2" if r > 150 else "MARRON"
                    elif g > 125:
                        return "ROSADO"
                    else:
                        return "ROJO 3"
                else:  # Green closer to red
                    if red_blue_diff < 60:
                        return "ROSADO*" if r > 160 else "ROJO"
                    else:
                        return "ROJO"
            
            # Green and blue are nearly equal
            else:
                if red_green_diff > 20:
                    return "ROJO" if (r >= 100 and (b < 60 or r >= 100)) else "MARRON"
                else:
                    return "GRIS"

        # GREEN DOMINANT ---------------------------------------------------
        elif g > r and g > b:
            green_blue_diff = diff(g, b)  # Difference between green and blue
            green_red_diff = diff(g, r)   # Difference between green and red

            # Pure green (red and blue are very low)
            if r < 65 and b < 65 and green_blue_diff > 60:
                return "VERDE"

            red_blue_diff = diff(r, b)  # Difference between red and blue

            # Red is more prominent than blue
            if r > b:
                if green_red_diff < green_blue_diff:  # Green mixed with red
                    return "AMARILLO" if (red_blue_diff >= 150 and green_red_diff <= 20) else "VERDE"
                else:
                    return "VERDE"
            
            # Blue is more prominent than red
            elif r < b:
                if green_blue_diff < green_red_diff:  # Green mixed with blue
                    return "TURQUESA" if green_blue_diff <= 20 else "VERDE"
                else:
                    return "VERDE"
            
            # Red and blue are nearly equal
            else:
                return "VERDE" if green_blue_diff > 10 else "GRIS"

        # BLUE DOMINANT ------------------------------------------------------
        elif b > r and b > g:
            blue_green_diff = diff(b, g)  # Difference between blue and green
            blue_red_diff = diff(b, r)    # Difference between blue and red

            # Pure blue (red and green are very low)
            if r < 65 and g < 65 and blue_green_diff > 60:
                return "AZUL"

            red_green_diff = diff(r, g)  # Difference between red and green

            # Red is more prominent than green
            if g < r:
                if blue_green_diff < red_green_diff:  # Blue mixed with green
                    return "TURQUESA" if blue_green_diff <= 20 else "CELESTE"
                else:
                    if red_green_diff <= 20:
                        return "LILA" if r >= 150 else "AZUL *************"
                    else:
                        return "AZUL"
            
            # Green is more prominent than red
            elif g > r:
                if blue_red_diff < red_green_diff:  # Blue mixed with red
                    if blue_red_diff <= 20:
                        if r > 150 and g < 75:
                            return "ROSADO FIUSHA"
                        elif r > 150:  # Fixed undefined variable "ir" to "r"
                            return "LILA"
                        else:
                            return "MORADO"
                    else:
                        return "MORADO"
                else:
                    if red_green_diff <= 20:
                        return "GRIS" if blue_green_diff <= 20 else "AZUL"
                    else:
                        return "AZUL"
            
            # Red and green are nearly equal
            else:
                if blue_green_diff > 20:
                    return "ROJO" if (r >= 100 and (b < 60 or r >= 100)) else "MARRON"
                else:
                    return "GRIS"

        # ALL CHANNELS NEARLY EQUAL ---------------------------------------
        else:
            return "GRIS"

    except Exception as e:
        return f"Error: Invalid input - {str(e)}"


if __name__ == "__main__":
    import sys

    # Set default RGB values if no arguments are provided
    default_r, default_g, default_b = 255, 0, 0  # Default to red
    
    # Parse command line arguments with fallback to defaults
    try:
        if len(sys.argv) == 4:
            r, g, b = sys.argv[1], sys.argv[2], sys.argv[3]
        else:
            print(f"No arguments provided. Using default RGB: ({default_r}, {default_g}, {default_b})")
            r, g, b = default_r, default_g, default_b
    except IndexError:
        print(f"Invalid arguments. Using default RGB: ({default_r}, {default_g}, {default_b})")
        r, g, b = default_r, default_g, default_b

    # Get and print the color result
    print(simpleColor(r, g, b))