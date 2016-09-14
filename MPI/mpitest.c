#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <mpi.h>

void control(int value);
void rand_sleep(void);

int main (int argc, char *argv[])
{
	int *array, size, rank, task_to_execute, task_complete = 1, work = 1;
	double init_time, lock_time, get_time, flush_time, put1_time, put2_time, unlock_time;
	double lock2_time, accumulate_time, return_time, complete_time, process_time;
	MPI_Win window;
	control(MPI_Init(&argc,&argv));
	control(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
	srand(time(NULL) + rank); // used to create random times for each task
	if (!rank) // only master allocates window
	{
		if (argc == 1)
		{
			printf("Ingrese la cantidad de tareas a ejecutar: ");
			control(!scanf("%d", &size));
		}
		else
		{
			if (argc == 3)
			{
				work = atoi(argv[1]);
				size = atoi(argv[2]);
			}
			else
			{
				printf("Error al ingresar parámetros.\n");
				MPI_Abort(MPI_COMM_WORLD, 1);
			}
		}
		if (size < 1)
		{
			printf("Error en la cantidad de tareas a ejecutar.\n");
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		array = (int*) calloc(size + 2, sizeof(int)); // array = size, next task to execute, task0, task1, ...., taskN (taskX => 0 not exectued, 1 done)
		array[0] = size;
		array[1] = 1;
	}
	control(MPI_Barrier(MPI_COMM_WORLD)); // to hold the slave process from beginning
	control(MPI_Win_create(!rank ? array : NULL, !rank ? sizeof(int) * (size + 2) : 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &window)); // only master exposes window
	// VER: por qué funciona sólo si la lock es exclusiva???? por qué no con una lock compartida???
	if (rank)
	{
		//control(MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, window)); // lock the window
		control(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, window)); // lock the window
		control(MPI_Get(&size, 1, MPI_INT, 0, 0, 1, MPI_INT, window)); // obtain window size (processes 1 to n)
		control(MPI_Win_unlock(0, window));
	}
	control(MPI_Barrier(MPI_COMM_WORLD));
	while(work)
	{
		//rand_sleep();
		// GET A TASK TO EXECUTE
		init_time = MPI_Wtime();
		control(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, window)); // lock the window
		lock_time = MPI_Wtime();
		control(MPI_Get(&task_to_execute, 1, MPI_INT, 0, 1, 1, MPI_INT, window));
		get_time = MPI_Wtime();
		control(MPI_Win_flush_local(0, window));
		flush_time = MPI_Wtime();
		if(task_to_execute > size)
		{
			control(MPI_Win_unlock(0, window));
			break;
		}
		// here it could be implemented the fact that it is possible for no tasks to be available to execute at the moment
		task_to_execute++; //assume the execution of the current task
		put1_time = MPI_Wtime();
		control(MPI_Put(&task_to_execute, 1, MPI_INT, 0, 1, 1, MPI_INT, window));
		put2_time = MPI_Wtime();
		control(MPI_Win_unlock(0, window));
		unlock_time = MPI_Wtime();
		// EXECUTE SELECTED TASK
		rand_sleep();
		process_time = MPI_Wtime();
		// UPDATE TASKS VECTOR
		control(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, window)); // lock the window
		lock2_time = MPI_Wtime();
		control(MPI_Accumulate(&task_complete, 1, MPI_INT, 0, task_to_execute, 1, MPI_INT, MPI_SUM, window));
		accumulate_time = MPI_Wtime();
		control(MPI_Win_unlock(0, window));
		return_time = MPI_Wtime();
		if (argc == 1)
		{
			printf(".");
		}
		complete_time = MPI_Wtime();
		printf("%d,LOCK,%f\n", rank, lock_time - init_time);
		printf("%d,GET,%f\n", rank, get_time - lock_time);
		printf("%d,FLUSH,%f\n", rank, flush_time - get_time);
		printf("%d,PUT,%f\n", rank, put2_time - put1_time);
		printf("%d,UNLOCK,%f\n", rank, unlock_time - put2_time);
		printf("%d,PROCESS,%f\n", rank, process_time - unlock_time);
		printf("%d,LOCK 2,%f\n", rank, lock2_time - process_time);
		printf("%d,ACCUMULATE,%f\n", rank, accumulate_time - lock2_time);
		printf("%d,UNLOCK 2,%f\n", rank, return_time - accumulate_time);
		printf("%d,TAKE,%f\n", rank, unlock_time - init_time);
		printf("%d,RETURN,%f\n", rank, return_time - process_time);
		printf("%d,COMPLETE,%f\n", rank, complete_time - init_time);
	}
	control(MPI_Barrier(MPI_COMM_WORLD)); // to hold the master process, who owns the array
	control(MPI_Win_free(&window));
	if (!rank && argc == 1) // only master allocates window
	{
		int iterator;
		printf("\nTasks array: ");
		for(iterator = 2; iterator < (size + 2); iterator++)
		{
			printf("%d ", array[iterator]);
		}
		printf("\n");
		free(array);
	}
	control(MPI_Finalize());
	return 0;
}

void control(int value)
{
	if(value != MPI_SUCCESS)
	{
		printf("MPI ERROR");
		exit(-1);
	}
}

void rand_sleep(void)
{
	//usleep(rand() % 999999); // sleep between 0 and 1 seconds
	/*time_t base_time, time_to_sleep;
	time_to_sleep = (rand() % 3) + 1;
	base_time = time(NULL);
	time_to_sleep += base_time;
	while(time_to_sleep > base_time)
	{
		base_time = time(NULL);
	}*/
	/* Sleep between 100ms and 500ms, 1ms each sleep */
	int iterator, limit;
	limit = (rand() % 401) + 100;
	for (iterator = 0; iterator < limit; iterator++)
	{
		usleep(1000);
	}
}
