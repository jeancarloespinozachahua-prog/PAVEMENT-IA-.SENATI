"""Constantes y utilidades del dominio PCI (pavimento rígido)."""

# Escala de condición (misma lógica que el prototipo HTML de la guía del reto).
def condicion(pci: float) -> str:
    if pci > 85:
        return "Bueno"
    if pci > 70:
        return "Satisfactorio"
    if pci > 55:
        return "Regular"
    if pci > 40:
        return "Pobre"
    if pci > 25:
        return "Muy pobre"
    if pci > 10:
        return "Grave"
    return "Fallado"


ORDEN_CONDICION = ["Bueno", "Satisfactorio", "Regular", "Pobre", "Muy pobre", "Grave", "Fallado"]

COLOR_CONDICION = {
    "Bueno": "#2e9e5b",
    "Satisfactorio": "#8bc34a",
    "Regular": "#f2c230",
    "Pobre": "#f08a24",
    "Muy pobre": "#e0562b",
    "Grave": "#b3261e",
    "Fallado": "#6b1a15",
}

# Patologías del pavimento rígido con el trazo SVG del prototipo (viewBox 0 0 100 100).
PATOLOGIAS = {
    "Grietas longitudinales": "M55 5 L48 35 58 60 50 95",
    "Grietas transversales": "M5 50 L35 45 65 55 95 48",
    "Grietas de esquina": "M5 40 L40 5",
    "Losa dividida": "M5 50 L50 40 95 60 M50 5 L50 95",
    "Despostillamiento": "M5 48 L20 40 30 56 45 42 60 57 80 40 95 50",
    "Escalonamiento": "M5 45 H50 V60 H95",
}

SEVERIDADES = ["Baja", "Media", "Alta"]
TRANSITOS = ["Bajo", "Medio", "Alto"]


def svg_patologia(nombre: str, lado: int = 90) -> str:
    """SVG inline de una patología (para mostrar en la interfaz)."""
    trazo = PATOLOGIAS[nombre]
    return (
        f'<svg width="{lado}" height="{lado}" viewBox="0 0 100 100" role="img" aria-label="{nombre}">'
        '<rect x="3" y="3" width="94" height="94" rx="4" fill="#aab5bd"/>'
        f'<path d="{trazo}" stroke="#263746" stroke-width="3" fill="none"/></svg>'
    )
