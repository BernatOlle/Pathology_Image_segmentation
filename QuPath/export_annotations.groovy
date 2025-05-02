import qupath.lib.images.servers.ImageServer
import qupath.lib.regions.RegionRequest
import qupath.lib.scripting.QP
import javax.imageio.ImageIO
import java.awt.image.BufferedImage
import java.awt.Color
import java.awt.geom.AffineTransform
import java.awt.Rectangle

// 1. Definir la ruta base del dataset
def basePath = "/mnt/work/users/bernat.olle/Dataset"

// 2. Obtener nombre base del archivo .mrxs (sin extensión)
def imageName = GeneralTools.getNameWithoutExtension(QP.getCurrentServer().getMetadata().getName())
println "Nombre de imagen: ${imageName}"

// 3. Extraer "R#" y "S#" del nombre (ej: "slide-2023-02-18T07-56-19-R3-S2"  "R3" y "S2")
def ratonPattern = ~/.*-(R\d+)-.*/ 
def seccionPattern = ~/.*-(S\d+).*/ 

def ratonMatcher = (imageName =~ ratonPattern)
def seccionMatcher = (imageName =~ seccionPattern)

def ratonID = ratonMatcher.matches() ? ratonMatcher[0][1] : "Unknown"
def seccionID = seccionMatcher.matches() ? seccionMatcher[0][1] : "Unknown"

println "ID de ratón extraído: ${ratonID}"
println "ID de sección extraído: ${seccionID}"

// 4. Crear directorios (img y mask dentro de Dataset/R#/S#)
def ratonDir = new File(basePath, ratonID)
def seccionDir = new File(ratonDir, seccionID)
def imgDir = new File(seccionDir, "img")
def maskDir = new File(seccionDir, "mask")
[seccionDir, imgDir, maskDir].each { it.mkdirs() }

// 5. Obtener el servidor de imagen y anotaciones
def server = QP.getCurrentServer()
def annotations = QP.getAnnotationObjects()

// 6. Definir el tamaño de los parches
def patchSize = 2048

// 7. Definir el stride (50% del tamaño del parche)
def stride = patchSize / 2

// 8. Obtener dimensiones de la imagen completa
def width = server.getWidth()
def height = server.getHeight()

// 9. Calcular el rectángulo que contiene todas las anotaciones (glomérulos)
double minX = Double.MAX_VALUE
double minY = Double.MAX_VALUE
double maxX = Double.MIN_VALUE
double maxY = Double.MIN_VALUE

// Verificar si hay anotaciones
if (annotations.isEmpty()) {
    println "¡No se encontraron anotaciones en la imagen!"
    return
}

// Encontrar los límites del rectángulo que contiene todas las anotaciones
annotations.each { annotation ->
    def roi = annotation.getROI()
    if (roi) {
        minX = Math.min(minX, roi.getBoundsX())
        minY = Math.min(minY, roi.getBoundsY())
        maxX = Math.max(maxX, roi.getBoundsX() + roi.getBoundsWidth())
        maxY = Math.max(maxY, roi.getBoundsY() + roi.getBoundsHeight())
    }
}

// 10. Añadir un margen alrededor del rectángulo (20% del tamaño del rectángulo)
def marginPercentage = 0.08
def rectWidth = maxX - minX
def rectHeight = maxY - minY
def marginX = rectWidth * marginPercentage
def marginY = rectHeight * marginPercentage

minX = Math.max(0, minX - marginX)
minY = Math.max(0, minY - marginY)
maxX = Math.min(width, maxX + marginX)
maxY = Math.min(height, maxY + marginY)

// Redondear a enteros
minX = minX as int
minY = minY as int
maxX = maxX as int
maxY = maxY as int

println "Rectángulo que contiene todos los glomérulos: (${minX}, ${minY}) - (${maxX}, ${maxY})"
println "Dimensiones del rectángulo con margen: ${maxX - minX}x${maxY - minY}"

// 11. Calcular cuántos parches se necesitan en el rectángulo con el stride
def startPatchX = Math.floor(minX / stride) as int
def startPatchY = Math.floor(minY / stride) as int
def endPatchX = Math.ceil(maxX / stride) as int
def endPatchY = Math.ceil(maxY / stride) as int

println "Procesando parches desde (${startPatchX}, ${startPatchY}) hasta (${endPatchX}, ${endPatchY})"

// Contador para parches guardados
def savedPatches = 0

// 12. Generar parches y máscaras solo para el área de interés
for (int patchY = startPatchY; patchY <= endPatchY; patchY++) {
    for (int patchX = startPatchX; patchX <= endPatchX; patchX++) {
        // Calcular coordenadas del parche con stride
        int x = (patchX * stride) as int
        int y = (patchY * stride) as int
        
        // Asegurar que no nos salimos de los límites de la imagen
        int w = Math.min(patchSize, width - x)
        int h = Math.min(patchSize, height - y)
        
        // Solo procesar parches completos de 2048x2048
        if (w == patchSize && h == patchSize) {
            try {
                // Verificar si el parche intersecta con el área de interés
                Rectangle patchRect = new Rectangle(x, y, patchSize, patchSize)
                Rectangle roiRect = new Rectangle(minX as int, minY as int, (maxX - minX) as int, (maxY - minY) as int)
                
                if (patchRect.intersects(roiRect)) {
                    // Crear RegionRequest para este parche
                    def request = RegionRequest.createInstance(server.getPath(), 1, x, y, patchSize, patchSize)
                    
                    // Guardar imagen del parche
                    def imgPatch = server.readRegion(request)
                    def imgFile = new File(imgDir, "${imageName}_${x}_${y}_img.png")
                    ImageIO.write(imgPatch, "PNG", imgFile)
                    
                    // Crear máscara para este parche
                    def mask = new BufferedImage(patchSize, patchSize, BufferedImage.TYPE_BYTE_GRAY)
                    def g2d = mask.createGraphics()
                    g2d.setColor(Color.WHITE)
                    
                    // Dibujar todas las anotaciones que intersectan con este parche
                    boolean hasAnnotations = false
                    annotations.each { annotation ->
                        def roi = annotation.getROI()
                        // Comprobar si el ROI intersecta con este parche
                        if (roi.getBoundsX() < x + patchSize && 
                            roi.getBoundsX() + roi.getBoundsWidth() > x &&
                            roi.getBoundsY() < y + patchSize && 
                            roi.getBoundsY() + roi.getBoundsHeight() > y) {
                            
                            // Obtener la forma y aplicar transformación para ajustar a coordenadas locales
                            def shape = roi.getShape()
                            def transform = new AffineTransform()
                            transform.translate(-x, -y)
                            def transformedShape = transform.createTransformedShape(shape)
                            
                            // Dibujar la forma en la máscara
                            g2d.fill(transformedShape)
                            hasAnnotations = true
                        }
                    }
                    
                    g2d.dispose()
                    
                    // Guardar la máscara (incluso si no tiene anotaciones)
                    def maskFile = new File(maskDir, "${imageName}_${x}_${y}_mask.png")
                    ImageIO.write(mask, "PNG", maskFile)
                    
                    savedPatches++
                    if (hasAnnotations) {
                        println "Guardado parche ${patchX},${patchY} (${x},${y}) - Con anotaciones"
                    } else {
                        println "Guardado parche ${patchX},${patchY} (${x},${y}) - Sin anotaciones (dentro del área de interés)"
                    }
                }
            } catch (Exception e) {
                println "Error procesando parche ${patchX},${patchY} (${x},${y}): ${e.getMessage()}"
                e.printStackTrace()
            }
        }
    }
}

print "¡Procesamiento completado! Se guardaron ${savedPatches} parches en la región de interés: ${basePath}/${ratonID}/"