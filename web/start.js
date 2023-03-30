var shell = require('shelljs');

function start() {
  if (shell) {

    shell.exec('run.sh', {async: true}, (code, stdout, stderr) => {
      console.log('Exit code:', code)
      console.log('Program output:', stdout)
      console.log('Program stderr:', stderr)
    })
  }
}