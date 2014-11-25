gulp    = require 'gulp'
gutil   = require 'gulp-util'
pandoc  = require 'gulp-pandoc'
connect = require 'gulp-connect'
less    = require 'gulp-less'

gulp.task 'markdown', ->
  gulp.src 'markdown/*.md'
    .pipe pandoc
      from: 'markdown'
      to: 'html5'
      ext: '.html'
      args: [
        '--smart'
        '-s'
        '--mathjax'
        # '--toc'
        '--css=style.css'
      ]
    .pipe gulp.dest 'html'
    .pipe connect.reload()

gulp.task 'less', ->
  gulp.src 'less/style.less'
    .pipe less()
    .pipe gulp.dest 'html'
    .pipe connect.reload()

gulp.task 'connect', ->
  connect.server
    root: 'html',
    livereload: true

gulp.task 'watch', ->
  gulp.watch ['markdown/*.md'], ['markdown']
  gulp.watch ['less/*.less'], ['less']

gulp.task 'default', ['less', 'markdown','connect', 'watch']

# pandoc LaTeX and CSS
# D3 SVG?
