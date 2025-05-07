// Script para vincular las anotaciones con la imagen MRXS en QuPath
// Guardar como "load_annotations.groovy" en el mismo directorio que el proyecto QuPath

import qupath.lib.images.servers.ImageServerProvider
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.projects.Project
import qupath.lib.objects.PathObjects
import qupath.lib.roi.ROIs
import qupath.lib.geom.Point2
import qupath.lib.io.PathIO
import qupath.lib.gui.QuPathGUI
import qupath.lib.gui.dialogs.Dialogs

// Obtener imagen abierta actualmente
def imageData = getCurrentImageData()

if (imageData == null) {
    Dialogs.showErrorMessage("Error", "¡No hay una imagen abierta! Por favor, abra primero el archivo MRXS")
    return
}

// Ruta al archivo .qpdata
def project = getProject()
def currentImageName = imageData.getServer().getMetadata().getName()
def qpdataPath = new File(project.getPath()).getParent() + File.separator + "qupath" + File.separator + currentImageName + "_glomeruli.qpdata"
def qpdataFile = new File(qpdataPath)

if (!qpdataFile.exists()) {
    Dialogs.showErrorMessage("Error", "¡No se encuentra el archivo de anotaciones!\nBuscando en: " + qpdataPath)
    return
}

// Cargar anotaciones
println("Cargando anotaciones desde " + qpdataPath)
def annotations = []

try {
    def json = new groovy.json.JsonSlurper().parseText(qpdataFile.text)
    
    // Procesar todas las anotaciones en el archivo
    json.annotations.each { annotationData ->
        def roiData = annotationData.roi
        if (roiData.name == "Polygon" && roiData.vertices) {
            def points = roiData.vertices.collect { new Point2(it.x, it.y) }
            if (points.size() >= 3) {
                def roi = ROIs.createPolygonROI(points, imageData.getServer().getPixelCalibration())
                def annotation = PathObjects.createAnnotationObject(roi)
                annotation.setName("Glomerulus")
                annotation.setColorRGB(-16776961) // Azul
                annotations.add(annotation)
            }
        }
    }
    
    // Añadir anotaciones a la imagen
    addObjects(annotations)
    println("Se importaron " + annotations.size() + " anotaciones de glomérulos")
    
    if (annotations.isEmpty()) {
        Dialogs.showWarningMessage("Advertencia", "No se encontraron anotaciones válidas en el archivo")
    } else {
        Dialogs.showInfoMessage("Éxito", "Se importaron " + annotations.size() + " anotaciones de glomérulos")
    }
    
} catch (Exception e) {
    Dialogs.showErrorMessage("Error", "Error al cargar anotaciones: " + e.getMessage())
    println("Error: " + e.getMessage())
    e.printStackTrace()
}