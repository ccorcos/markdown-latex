# Markdown LaTeX Editor

This gulpfile will help you to generate markdown files with LaTeX.

## Getting Started

If you havent installed gulp already

    $ npm install -g gulp

After you `git clone` this repository

    $ npm install
    $ gulp

And now you're good to go. Create markdown files in the `markdown` directory. You can view their compiled html at `localhost:8080` and they should live-reload when you save a file.

You can style the html with less, and you can edit the `pandoc` command in the `gulpfile.coffee`, for example toggling `--toc` would toggle the table of contents showing up at the top of the page.
