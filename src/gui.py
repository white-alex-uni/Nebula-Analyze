import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from PIL import Image, ImageTk

from src.proceed import (
    SUPPORTED_EXTENSIONS,
    load_image,
    load_images_from_folder,
    save_image,
)
from src.run import ALGORITHMS, timer


class SpaceAnalyzeApp:
    """Графічний інтерфейс (Tkinter) для обробки та аналізу зображень космічних об'єктів."""

    def __init__(self, root: tk.Tk, output_dir: str | Path = r".\output") -> None:
        self.root = root
        self.root.title("Аналіз космічних об'єктів — Обробка зображень")
        self.root.geometry("1150x720")
        self.root.minsize(850, 540)

        self.output_dir = Path(output_dir)

        # Список завантажених елементів: [(Image, Path), ...]
        self.loaded_items: list[tuple[Image.Image, Path]] = []
        # Словник результатів обробки за індексом: {idx: (Processed_Image, Saved_Path)}
        self.processed_items: dict[int, tuple[Image.Image, Path | None]] = {}
        self.current_index: int = 0
        self.is_processing: bool = False

        # Збереження посилань на PhotoImage, щоб їх не видалив garbage collector
        self._orig_photo: ImageTk.PhotoImage | None = None
        self._proc_photo: ImageTk.PhotoImage | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        # Верхня панель керування
        control_frame = ttk.Frame(self.root, padding=(12, 10))
        control_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_load_file = ttk.Button(
            control_frame,
            text="Завантажити зображення",
            command=self._on_load_file,
        )
        self.btn_load_file.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_load_folder = ttk.Button(
            control_frame,
            text="Завантажити папку",
            command=self._on_load_folder,
        )
        self.btn_load_folder.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(control_frame, text="Алгоритм:").pack(side=tk.LEFT, padx=(0, 6))

        algo_names = list(ALGORITHMS.keys())
        self.algo_var = tk.StringVar(value=algo_names[0] if algo_names else "")
        self.algo_combo = ttk.Combobox(
            control_frame,
            textvariable=self.algo_var,
            values=algo_names,
            state="readonly",
            width=40,
        )
        self.algo_combo.pack(side=tk.LEFT, padx=(0, 12))

        self.btn_run = ttk.Button(
            control_frame,
            text="Виконати обробку",
            command=self._on_run_algorithm,
            state=tk.DISABLED,
        )
        self.btn_run.pack(side=tk.LEFT, padx=(0, 10))

        # Панель навігації по завантажених зображеннях (актуально для папки)
        nav_frame = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        nav_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_prev = ttk.Button(
            nav_frame,
            text="◀ Попереднє",
            command=self._on_prev_image,
            state=tk.DISABLED,
            width=14,
        )
        self.btn_prev.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_next = ttk.Button(
            nav_frame,
            text="Наступне ▶",
            command=self._on_next_image,
            state=tk.DISABLED,
            width=14,
        )
        self.btn_next.pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(nav_frame, text="Файл:").pack(side=tk.LEFT, padx=(0, 6))
        self.file_var = tk.StringVar(value="Не обрано")
        self.file_combo = ttk.Combobox(
            nav_frame,
            textvariable=self.file_var,
            state="disabled",
            width=55,
        )
        self.file_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.file_combo.bind("<<ComboboxSelected>>", self._on_select_file_combo)

        # Основна область відображення (Оригінал та Оброблене зображення)
        view_frame = ttk.Frame(self.root, padding=(12, 4, 12, 8))
        view_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        view_frame.columnconfigure(0, weight=1)
        view_frame.columnconfigure(1, weight=1)
        view_frame.rowconfigure(0, weight=1)

        # Ліва панель — Оригінальне зображення
        orig_group = ttk.LabelFrame(view_frame, text="Оригінальне зображення", padding=6)
        orig_group.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.orig_canvas = tk.Canvas(orig_group, bg="#1e1e1e", highlightthickness=0)
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)

        # Права панель — Оброблене зображення
        proc_group = ttk.LabelFrame(view_frame, text="Результат обробки", padding=6)
        proc_group.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.proc_canvas = tk.Canvas(proc_group, bg="#1e1e1e", highlightthickness=0)
        self.proc_canvas.pack(fill=tk.BOTH, expand=True)

        self.orig_canvas.bind("<Configure>", lambda _e: self._redraw_canvases())
        self.proc_canvas.bind("<Configure>", lambda _e: self._redraw_canvases())

        # Нижня панель статусу та інформації про файл
        status_frame = ttk.Frame(self.root, padding=(12, 6))
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar(
            value="Завантажте одне зображення або папку із зображеннями для початку роботи."
        )
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.info_var = tk.StringVar(
            value="Ширина × Висота: —  |  Оригінал: —  |  Оброблений: —"
        )
        self.info_label = ttk.Label(status_frame, textvariable=self.info_var, anchor="e")
        self.info_label.pack(side=tk.RIGHT, padx=(12, 0))

    def _on_load_file(self) -> None:
        if self.is_processing:
            return
        ext_pattern = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        file_path = filedialog.askopenfilename(
            title="Оберіть зображення",
            initialdir=str(Path("./images").resolve()) if Path("./images").is_dir() else ".",
            filetypes=[
                ("Зображення", ext_pattern),
                ("Усі файли", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            img, path = load_image(file_path)
            self.loaded_items = [(img, path)]
            self.processed_items.clear()
            self.current_index = 0
            self._update_navigation_state()
            self._redraw_canvases()
            self.status_var.set(
                f"Завантажено: {path.name} ({img.size[0]}x{img.size[1]}, формат: {img.format or path.suffix})"
            )
        except Exception as e:
            messagebox.showerror("Помилка завантаження", f"Не вдалося відкрити файл:\n{e}")

    def _on_load_folder(self) -> None:
        if self.is_processing:
            return
        folder_path = filedialog.askdirectory(
            title="Оберіть папку із зображеннями",
            initialdir=str(Path("./images").resolve()) if Path("./images").is_dir() else ".",
        )
        if not folder_path:
            return

        try:
            items = load_images_from_folder(folder_path)
            if not items:
                messagebox.showwarning(
                    "Порожня папка",
                    "У вибраній папці не знайдено підтримуваних зображень.",
                )
                return

            self.loaded_items = items
            self.processed_items.clear()
            self.current_index = 0
            self._update_navigation_state()
            self._redraw_canvases()
            self.status_var.set(
                f"Завантажено папку '{Path(folder_path).name}': знайдено зображень — {len(items)}."
            )
        except Exception as e:
            messagebox.showerror("Помилка завантаження папки", f"Не вдалося прочитати папку:\n{e}")

    def _update_navigation_state(self) -> None:
        total = len(self.loaded_items)
        has_items = total > 0

        self.btn_run.config(state=tk.NORMAL if (has_items and not self.is_processing) else tk.DISABLED)

        if has_items:
            names = [f"[{i + 1}/{total}] {path.name}" for i, (_, path) in enumerate(self.loaded_items)]
            self.file_combo.config(values=names, state="readonly" if not self.is_processing else "disabled")
            self.file_combo.current(self.current_index)
            self.btn_prev.config(
                state=tk.NORMAL if (self.current_index > 0 and not self.is_processing) else tk.DISABLED
            )
            self.btn_next.config(
                state=tk.NORMAL if (self.current_index < total - 1 and not self.is_processing) else tk.DISABLED
            )
        else:
            self.file_combo.config(values=[], state="disabled")
            self.file_var.set("Не обрано")
            self.btn_prev.config(state=tk.DISABLED)
            self.btn_next.config(state=tk.DISABLED)

    def _on_prev_image(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self._update_navigation_state()
            self._redraw_canvases()

    def _on_next_image(self) -> None:
        if self.current_index < len(self.loaded_items) - 1:
            self.current_index += 1
            self.current_index = min(self.current_index, len(self.loaded_items) - 1)
            self._update_navigation_state()
            self._redraw_canvases()

    def _on_select_file_combo(self, _event: tk.Event) -> None:
        idx = self.file_combo.current()
        if 0 <= idx < len(self.loaded_items):
            self.current_index = idx
            self._update_navigation_state()
            self._redraw_canvases()

    def _on_run_algorithm(self) -> None:
        if not self.loaded_items or self.is_processing:
            return

        algo_name = self.algo_var.get()
        if algo_name not in ALGORITHMS:
            messagebox.showerror("Помилка", "Оберіть алгоритм зі списку.")
            return

        runner_fn, suffix = ALGORITHMS[algo_name]

        self.is_processing = True
        self.btn_load_file.config(state=tk.DISABLED)
        self.btn_load_folder.config(state=tk.DISABLED)
        self.algo_combo.config(state="disabled")
        self._update_navigation_state()

        thread = threading.Thread(
            target=self._process_worker,
            args=(algo_name, runner_fn, suffix),
            daemon=True,
        )
        thread.start()

    def _process_worker(self, algo_name: str, runner_fn: callable, suffix: str) -> None:
        total = len(self.loaded_items)
        saved_count = 0

        for idx, (orig_img, orig_path) in enumerate(self.loaded_items):
            self.root.after(
                0,
                self.status_var.set,
                f"Обробка [{idx + 1}/{total}]: {orig_path.name} ({algo_name})...",
            )
            try:
                result_img = runner_fn(orig_img)
                saved_path = save_image(
                    img=result_img,
                    original_path=orig_path,
                    output_path=self.output_dir,
                    suffix=suffix,
                )
                self.processed_items[idx] = (result_img, saved_path)
                if saved_path is not None:
                    saved_count += 1

                # Показуємо щойно оброблене зображення у вікні
                self.root.after(0, self._show_processed_index, idx, total)
            except Exception as e:
                self.root.after(
                    0,
                    lambda err=e, name=orig_path.name: messagebox.showerror(
                        "Помилка обробки",
                        f"Помилка під час обробки файлу {name}:\n{err}",
                    ),
                )

        self.root.after(0, self._on_processing_complete, algo_name, saved_count, total)

    def _show_processed_index(self, idx: int, total: int) -> None:
        self.current_index = idx
        if self.loaded_items:
            names = [f"[{i + 1}/{total}] {path.name}" for i, (_, path) in enumerate(self.loaded_items)]
            self.file_combo.config(values=names)
            self.file_combo.current(self.current_index)
        self._redraw_canvases()

    def _on_processing_complete(self, algo_name: str, saved_count: int, total: int) -> None:
        self.is_processing = False
        self.btn_load_file.config(state=tk.NORMAL)
        self.btn_load_folder.config(state=tk.NORMAL)
        self.algo_combo.config(state="readonly")
        self._update_navigation_state()
        self._redraw_canvases()

        last_times = list(timer.last_durations.values())
        time_info = f" (останній кадр: {last_times[-1]:.3f} с)" if last_times else ""
        self.status_var.set(
            f"Готово! Алгоритм '{algo_name}': збережено {saved_count}/{total} у папку '{self.output_dir}'{time_info}."
        )

    @staticmethod
    def _format_file_size(file_path: Path | None) -> str:
        if file_path is None or not file_path.is_file():
            return "—"
        size_bytes = file_path.stat().st_size
        if size_bytes < 1024:
            return f"{size_bytes} Б"
        size_kb = size_bytes / 1024.0
        if size_kb < 1024:
            return f"{size_kb:.1f} КБ"
        size_mb = size_kb / 1024.0
        return f"{size_mb:.2f} МБ"

    def _update_image_info(self) -> None:
        if not self.loaded_items or not (0 <= self.current_index < len(self.loaded_items)):
            self.info_var.set("Ширина × Висота: —  |  Оригінал: —  |  Оброблений: —")
            return

        orig_img, orig_path = self.loaded_items[self.current_index]
        width, height = orig_img.size
        orig_size_str = self._format_file_size(orig_path)

        proc_size_str = "—"
        if self.current_index in self.processed_items:
            _, saved_path = self.processed_items[self.current_index]
            proc_size_str = self._format_file_size(saved_path)

        self.info_var.set(
            f"Ширина × Висота: {width} × {height} px  |  Оригінал: {orig_size_str}  |  Оброблений: {proc_size_str}"
        )

    def _redraw_canvases(self) -> None:
        self.orig_canvas.delete("all")
        self.proc_canvas.delete("all")
        self._update_image_info()

        if not self.loaded_items or not (0 <= self.current_index < len(self.loaded_items)):
            self._draw_placeholder(self.orig_canvas, "Зображення не завантажено")
            self._draw_placeholder(self.proc_canvas, "Очікування результату обробки")
            return

        orig_img, _ = self.loaded_items[self.current_index]
        self._orig_photo = self._render_image_on_canvas(self.orig_canvas, orig_img)

        if self.current_index in self.processed_items:
            proc_img, _ = self.processed_items[self.current_index]
            self._proc_photo = self._render_image_on_canvas(self.proc_canvas, proc_img)
        else:
            self._draw_placeholder(
                self.proc_canvas,
                "Натисніть «Виконати обробку»,\nщоб переглянути та зберегти результат",
            )

    @staticmethod
    def _draw_placeholder(canvas: tk.Canvas, text: str) -> None:
        w = max(canvas.winfo_width(), 200)
        h = max(canvas.winfo_height(), 200)
        canvas.create_text(
            w // 2,
            h // 2,
            text=text,
            fill="#9e9e9e",
            font=("Segoe UI", 11),
            justify=tk.CENTER,
        )

    @staticmethod
    def _render_image_on_canvas(canvas: tk.Canvas, img: Image.Image) -> ImageTk.PhotoImage | None:
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()
        if canvas_w <= 10 or canvas_h <= 10:
            return None

        img_w, img_h = img.size
        if img_w <= 0 or img_h <= 0:
            return None

        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = img.convert("RGB").resize((new_w, new_h), Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(resized)
        canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo, anchor=tk.CENTER)
        return photo


def launch_gui(output_dir: str | Path = r".\output") -> None:
    root = tk.Tk()
    SpaceAnalyzeApp(root, output_dir=output_dir)
    root.mainloop()
