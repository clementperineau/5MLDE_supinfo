import { Component } from '@angular/core';

declare function start(): any;

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = '5MLRE_Front';

  constructor() {}
  runPrediction() {
    console.log('AppComponent runPrediction');
    start()
  }
}