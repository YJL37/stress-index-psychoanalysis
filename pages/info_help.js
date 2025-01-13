const slider = document.querySelector('.slider');
const slides = document.querySelectorAll('.slide');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');

let currentSlide = 0;

// Function to update slider position
function updateSlider() {
    const offset = -currentSlide * 100; // Calculate the translateX value
    slider.style.transform = `translateX(${offset}%)`;
}

// Navigate to the previous slide
prevBtn.addEventListener('click', () => {
    currentSlide = (currentSlide - 1 + slides.length) % slides.length;
    updateSlider();
});

// Navigate to the next slide
nextBtn.addEventListener('click', () => {
    currentSlide = (currentSlide + 1) % slides.length;
    updateSlider();
});

// Initialize the slider position
updateSlider();
