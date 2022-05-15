
    (function() {
	var canvas = document.querySelector("#canvas");
	var context = canvas.getContext("2d");

	canvas.width = 280;
	canvas.height = 280;


	var Mouse = {x:0, y:0};
	var lastMouse = {x:0, y:0};

	context.fillStyle = "white";
	context.fillRect(0, 0, canvas.width, canvas.height);
	context.color = "black";
	context.lineWidth = 3;
    context.lineJoin = context.lineCap = 'round';

	canvas.addEventListener("mousemove", function(e) {
		lastMouse.x = Mouse.x;
		lastMouse.y = Mouse.y;

        var rect = canvas.getBoundingClientRect();

		Mouse.x = e.pageX - rect.left;
		Mouse.y = e.pageY - rect.top;

	}, false);



	var onPaint = function() {
        console.log("ne5dem")
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;

		context.beginPath();
		context.moveTo(lastMouse.x, lastMouse.y);
		context.lineTo(Mouse.x,Mouse.y );
		context.closePath();
		context.stroke();

	};
    canvas.addEventListener("mousedown", function(e) {

		canvas.addEventListener("mousemove", onPaint, false);
	}, false);

	canvas.addEventListener("mouseup", function() {
		canvas.removeEventListener("mousemove", onPaint, false);
	}, false);


    $("#clearButton").on("click", function() {
			context.clearRect( 0, 0, 280, 280 );
			context.fillStyle="white";
			context.fillRect(0,0,canvas.width,canvas.height);
		});

    $("#nextButton").on("click", function() {


	   			var canvasObj = document.getElementById("canvas");
                var canvasObj2 = document.getElementById("canvas2");
                var canvasObj3 = document.getElementById("canvas3");
                var canvasObj4 = document.getElementById("canvas4");

	   			var img = canvasObj.toDataURL('image/png');
                var img2 = canvasObj2.toDataURL('image/png');
                var img3 = canvasObj3.toDataURL('image/png');
                var img4 = canvasObj4.toDataURL('image/png');
	   			$.ajax({
	   				type: "POST",
	   				url: "/test",
	   				data: {image1:img ,image2: img2,image3:img3,image4:img4},
                    traditional: true,
	   				success: function(data){
	   					$('#result').text('Predicted Output: ' + data);
	   				}
	   			});



                context.clearRect( 0, 0, 280, 280 );
			    context.fillStyle="white";
			    context.fillRect(0,0,canvas.width,canvas.height);
		});
}());


$(document).ready(function(){

var current_fs, next_fs, previous_fs; //fieldsets
var opacity;

$(".next").click(function(){

current_fs = $(this).parent();
next_fs = $(this).parent().next();
$("#progressbar li").eq($("fieldset").index(next_fs)).addClass("active");
next_fs.show();
current_fs.animate({opacity: 0}, {
step: function(now) {
opacity = 1 - now;
current_fs.css({
'display': 'none',
'position': 'relative'
});
next_fs.css({'opacity': opacity});
},
duration: 600
});
});
$(".previous").click(function(){
    console.log('test')
current_fs = $(this).parent();
previous_fs = $(this).parent().prev();
$("#progressbar li").eq($("fieldset").index(current_fs)).removeClass("active");
previous_fs.show();
current_fs.animate({opacity: 0}, {
step: function(now) {
opacity = 1 - now;
current_fs.css({
'display': 'none',
'position': 'relative'
});
previous_fs.css({'opacity': opacity});
},
duration: 600
});
});

$('.radio-group .radio').click(function(){
$(this).parent().find('.radio').removeClass('selected');
$(this).addClass('selected');
});

$(".submit").click(function(){
return false;
})

});



    (function() {
	var canvas = document.querySelector("#canvas2");
	var context = canvas.getContext("2d");

	canvas.width = 280;
	canvas.height = 280;


	var Mouse = {x:0, y:0};
	var lastMouse = {x:0, y:0};

	context.fillStyle = "white";
	context.fillRect(0, 0, canvas.width, canvas.height);
	context.color = "black";
	context.lineWidth = 3;
    context.lineJoin = context.lineCap = 'round';

	canvas.addEventListener("mousemove", function(e) {
		lastMouse.x = Mouse.x;
		lastMouse.y = Mouse.y;

        var rect = canvas.getBoundingClientRect();

		Mouse.x = e.pageX - rect.left;
		Mouse.y = e.pageY - rect.top;

	}, false);



	var onPaint = function() {
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;

		context.beginPath();
		context.moveTo(lastMouse.x, lastMouse.y);
		context.lineTo(Mouse.x,Mouse.y );
		context.closePath();
		context.stroke();

	};
    canvas.addEventListener("mousedown", function(e) {

		canvas.addEventListener("mousemove", onPaint, false);
	}, false);

	canvas.addEventListener("mouseup", function() {
		canvas.removeEventListener("mousemove", onPaint, false);
	}, false);


    $("#clearButton2").on("click", function() {
			context.clearRect( 0, 0, 280, 280 );
			context.fillStyle="white";
			context.fillRect(0,0,canvas.width,canvas.height);
		});


}());


    (function() {
	var canvas = document.querySelector("#canvas3");
	var context = canvas.getContext("2d");

	canvas.width = 280;
	canvas.height = 280;


	var Mouse = {x:0, y:0};
	var lastMouse = {x:0, y:0};

	context.fillStyle = "white";
	context.fillRect(0, 0, canvas.width, canvas.height);
	context.color = "black";
	context.lineWidth = 3;
    context.lineJoin = context.lineCap = 'round';

	canvas.addEventListener("mousemove", function(e) {
		lastMouse.x = Mouse.x;
		lastMouse.y = Mouse.y;

        var rect = canvas.getBoundingClientRect();

		Mouse.x = e.pageX - rect.left;
		Mouse.y = e.pageY - rect.top;

	}, false);



	var onPaint = function() {
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;

		context.beginPath();
		context.moveTo(lastMouse.x, lastMouse.y);
		context.lineTo(Mouse.x,Mouse.y );
		context.closePath();
		context.stroke();

	};
    canvas.addEventListener("mousedown", function(e) {

		canvas.addEventListener("mousemove", onPaint, false);
	}, false);

	canvas.addEventListener("mouseup", function() {
		canvas.removeEventListener("mousemove", onPaint, false);
	}, false);


    $("#clearButton3").on("click", function() {
			context.clearRect( 0, 0, 280, 280 );
			context.fillStyle="white";
			context.fillRect(0,0,canvas.width,canvas.height);
		});


}());


    (function() {
	var canvas = document.querySelector("#canvas4");
	var context = canvas.getContext("2d");

	canvas.width = 280;
	canvas.height = 280;


	var Mouse = {x:0, y:0};
	var lastMouse = {x:0, y:0};

	context.fillStyle = "white";
	context.fillRect(0, 0, canvas.width, canvas.height);
	context.color = "black";
	context.lineWidth = 3;
    context.lineJoin = context.lineCap = 'round';

	canvas.addEventListener("mousemove", function(e) {
		lastMouse.x = Mouse.x;
		lastMouse.y = Mouse.y;

        var rect = canvas.getBoundingClientRect();

		Mouse.x = e.pageX - rect.left;
		Mouse.y = e.pageY - rect.top;

	}, false);



	var onPaint = function() {
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;

		context.beginPath();
		context.moveTo(lastMouse.x, lastMouse.y);
		context.lineTo(Mouse.x,Mouse.y );
		context.closePath();
		context.stroke();

	};
    canvas.addEventListener("mousedown", function(e) {

		canvas.addEventListener("mousemove", onPaint, false);
	}, false);

	canvas.addEventListener("mouseup", function() {
		canvas.removeEventListener("mousemove", onPaint, false);
	}, false);


    $("#clearButton4").on("click", function() {
			context.clearRect( 0, 0, 280, 280 );
			context.fillStyle="white";
			context.fillRect(0,0,canvas.width,canvas.height);
		});


}());
